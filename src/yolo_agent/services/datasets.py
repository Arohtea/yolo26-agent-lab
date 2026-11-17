"""协调短数据库事务、平台文件接管与白名单异步任务。"""

import hashlib
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from fastapi import UploadFile
from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.serializers import JSONSerializer
from sqlalchemy import Engine, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import AnnotationRun
from yolo_agent.models.dataset import Dataset, DatasetImage, DatasetTask, DatasetVersion, VersionImage, new_id, utc_now
from yolo_agent.schemas.dataset import AnalysisCreate, ImageGroups, VersionCreate, UploadCreate
from yolo_agent.services.inspector import FORMATS
from yolo_agent.settings import Settings
from yolo_agent.workers.queue import enqueue_dataset_task, redis_connection

logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """可向调用者解释的业务错误。"""

    def __init__(self, status_code: int, message: str, task_id: str | None = None):
        """保存 HTTP 状态及可追踪任务标识。

        :param status_code: 对应的 HTTP 状态码。
        :param message: 不包含凭据的业务说明。
        :param task_id: 已创建的后台任务标识。
        :returns: 无。
        """
        super().__init__(message)
        self.status_code = status_code
        self.task_id = task_id


def owned_path(settings: Settings, relative: str) -> Path:
    """读取平台记录时再次约束路径，禁止越过存储根目录。

    :param settings: 平台存储配置。
    :param relative: 平台生成并保存在数据库的相对路径。
    :returns: 根目录内的绝对路径。
    :raises DatasetError: 文件路径越界。
    """
    root = settings.storage_dir.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise DatasetError(409, "存储记录越界，拒绝读取")
    return path


def fail_dataset_task(engine: Engine, task_id: str, message: str) -> None:
    """仅终结仍在执行的任务，不覆盖已经完成的版本与报告。

    :param engine: 当前进程数据库引擎。
    :param task_id: 平台任务 ID。
    :param message: 失败原因。
    :returns: 无。
    """
    with Session(engine) as session, session.begin():
        task = session.get(DatasetTask, task_id)
        if task is None or task.status not in {"queued", "running", "canceling"}:
            return
        task.status = "stopped" if task.stop_requested else "failed"
        task.error, task.finished_at = ("操作者终止任务" if task.stop_requested else message[:4000]), utc_now()
        session.execute(update(DatasetImage).where(DatasetImage.import_task_id == task_id, DatasetImage.status == "pending").values(status="failed", error="导入任务中断，图片尚未完成检查"))
        # 队列记录丢失或取消时未必触发 RQ 回调，推理批次必须与失败任务一起收敛。
        if task.kind == "auto_annotate":
            session.execute(update(AnnotationRun).where(AnnotationRun.id == task.parameters.get("run_id"), AnnotationRun.status.in_(["queued", "running"])).values(status=task.status, error=task.error or message[:4000]))


class DatasetService:
    """API 使用同步线程池执行短事务与有界文件复制。"""

    def __init__(self, engine: Engine, settings: Settings):
        """绑定当前 API 进程资源。

        :param engine: 进程独享数据库引擎。
        :param settings: 上传限制与存储配置。
        :returns: 无。
        """
        self.engine = engine
        self.settings = settings

    def _require_idle(self, session: Session, dataset_id: str, *, excluding: str | None = None) -> None:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None or dataset.deleted_at:
            raise DatasetError(404, "数据集不存在")
        active = session.scalar(select(DatasetTask.id).where(DatasetTask.dataset_id == dataset_id, DatasetTask.status.in_(["uploading", "queued", "running", "canceling"]), DatasetTask.id != excluding if excluding else True).limit(1))
        if active:
            raise DatasetError(409, "该数据集已有任务，请等待完成后再提交", active)

    def delete_asset(self, kind: str, identity: str, confirm_name: str) -> dict:
        """删除工作区资产，追加审计；冻结版本和模型历史继续可读。"""
        from yolo_agent.models.agent import AgentRun
        from yolo_agent.models.experiment import Experiment, TrainingTask

        with Session(self.engine) as session, session.begin():
            asset = session.get(Dataset if kind == "dataset" else DatasetImage, identity)
            if asset is None or asset.deleted_at:
                raise DatasetError(404, "待删除对象不存在或已经删除")
            name = asset.name if kind == "dataset" else asset.original_name
            if confirm_name != name:
                raise DatasetError(422, "确认名称与待删除对象不一致")
            dataset_id = asset.id if kind == "dataset" else asset.dataset_id
            self._require_idle(session, dataset_id)
            active_run = session.scalar(select(AgentRun.id).where(AgentRun.dataset_id == dataset_id, AgentRun.status.not_in(["completed", "stopped", "failed"])).limit(1))
            active_training = session.scalar(select(TrainingTask.id).join(Experiment).join(DatasetVersion, DatasetVersion.id == Experiment.dataset_version_id).where(DatasetVersion.dataset_id == dataset_id, TrainingTask.status.in_(["queued", "running", "canceling"])).limit(1))
            if active_run or active_training:
                raise DatasetError(409, "该数据集还有活跃流程，请先结束整个流程后再删除")
            timestamp = utc_now()
            asset.deleted_at = timestamp
            session.add(DatasetTask(dataset_id=dataset_id, kind=f"delete_{kind}", status="completed", total=1, processed=1,
                parameters={"asset_id": identity, "name": name, "source": "operator_workspace"},
                result={"deleted_from_workspace": True, "frozen_history_preserved": True}, started_at=timestamp, finished_at=timestamp))
        return {"deleted": True, "history_preserved": True}

    def begin_upload(self, dataset_id: str, data: UploadCreate) -> DatasetTask:
        """在大请求前登记身份，取消与接管文件通过同一任务事务仲裁。"""
        maximum = self.settings.max_upload_files if data.kind == "import" else self.settings.max_dataset_images * 2 + 200
        if data.total > maximum:
            raise DatasetError(413, f"本批文件数量超过上限 {maximum}")
        try:
            with Session(self.engine, expire_on_commit=False) as session, session.begin():
                self._require_idle(session, dataset_id)
                task = DatasetTask(dataset_id=dataset_id, kind=data.kind, status="uploading", total=data.total)
                session.add(task)
            return task
        except IntegrityError as error:
            raise DatasetError(409, "该数据集已有上传或处理任务") from error

    def require_upload(self, session: Session, identity: str, dataset_id: str, kind: str) -> DatasetTask:
        task = session.get(DatasetTask, identity)
        if task is None or task.dataset_id != dataset_id or task.kind != kind:
            raise DatasetError(404, "上传任务不存在或不属于该数据集")
        if task.status != "uploading":
            raise DatasetError(409, "上传已取消或已接收，不能继续提交文件", identity)
        return task

    def cancel_upload(self, identity: str) -> DatasetTask:
        """只取消未入队上传；一旦接收成功，不能误报为取消成功。"""
        with Session(self.engine, expire_on_commit=False) as session, session.begin():
            changed = session.execute(update(DatasetTask).where(DatasetTask.id == identity, DatasetTask.status == "uploading").values(status="stopped", finished_at=utc_now()))
            task = session.get(DatasetTask, identity)
            if task is None:
                raise DatasetError(404, "上传任务不存在")
            if not changed.rowcount and task.status != "stopped":
                raise DatasetError(409, "文件已接收并进入后台处理，上传阶段已结束", identity)
            return task

    def copy_upload(self, source, destination, identity: str | None, dataset_id: str, kind: str) -> None:
        """复用标准库块复制，每 8 MiB 检查一次取消，内存不随文件增长。"""
        source.seek(0)
        with destination.open("xb") as output:
            while True:
                if identity:
                    with Session(self.engine) as session:
                        self.require_upload(session, identity, dataset_id, kind)
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

    def _submit(self, task_id: str) -> DatasetTask:
        try:
            enqueue_dataset_task(self.settings, task_id)
        except Exception as exc:
            logger.exception("队列提交失败", extra={"task_id": task_id})
            fail_dataset_task(self.engine, task_id, f"队列提交失败：{type(exc).__name__}")
            raise DatasetError(503, "计算队列不可用，任务已保存为失败状态", task_id) from exc
        return self.get_task(task_id)

    def import_images(self, dataset_id: str, files: list[UploadFile], metadata: dict[str, ImageGroups], *, upload_id: str | None = None) -> DatasetTask:
        """接管受限数量的上传文件，解码与指纹计算留给 Worker。

        :param dataset_id: 数据集标识。
        :param files: multipart 上传的文件对象。
        :param metadata: 按上传相对文件名映射的采集分组。
        :returns: 已入队的持久化导入任务。
        :raises DatasetError: 数据集忙、上传非法或队列不可用。
        """
        settings = self.settings
        if not files or len(files) > settings.max_upload_files:
            raise DatasetError(413, f"每次允许上传 1 至 {settings.max_upload_files} 张图片")
        names = []
        for file in files:
            name = file.filename or ""
            path = PurePosixPath(name)
            if not name or len(name) > 500 or any(ord(char) < 32 for char in name) or "\\" in name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
                raise DatasetError(422, "文件名必须是无越界片段的相对路径")
            if path.suffix.lower() not in FORMATS:
                raise DatasetError(422, f"不支持的图片扩展名：{path.suffix}")
            if file.size is None or file.size <= 0:
                raise DatasetError(422, "图片文件不能为空")
            names.append(name)
        if len(names) != len(set(names)):
            raise DatasetError(422, "同一次上传的相对文件名必须唯一")
        if set(metadata) - set(names):
            raise DatasetError(422, "metadata 包含未在本次上传中出现的文件名")
        with Session(self.engine) as session:
            self._require_idle(session, dataset_id, excluding=upload_id)
            if upload_id:
                self.require_upload(session, upload_id, dataset_id, "import")

        task_id = upload_id or new_id()
        directory = owned_path(settings, f"images/{task_id}")
        directory.mkdir(parents=True)
        images = []
        committed = False
        try:
            for file, name in zip(files, names, strict=True):
                image_id = new_id()
                relative = f"images/{task_id}/{image_id}{PurePosixPath(name).suffix.lower()}"
                self.copy_upload(file.file, owned_path(settings, relative), upload_id, dataset_id, "import")
                images.append(DatasetImage(
                    id=image_id, dataset_id=dataset_id, import_task_id=task_id,
                    original_name=name, storage_path=relative, size_bytes=file.size,
                    **metadata.get(name, ImageGroups()).model_dump(),
                ))
            with Session(self.engine) as session, session.begin():
                self._require_idle(session, dataset_id, excluding=upload_id)
                if upload_id:
                    changed = session.execute(update(DatasetTask).where(DatasetTask.id == upload_id, DatasetTask.dataset_id == dataset_id, DatasetTask.kind == "import", DatasetTask.status == "uploading").values(status="queued", total=len(images)))
                    if changed.rowcount != 1:
                        raise DatasetError(409, "上传已取消，文件不会进入导入队列", upload_id)
                count = session.scalar(select(func.count()).select_from(DatasetImage).where(DatasetImage.dataset_id == dataset_id, DatasetImage.deleted_at.is_(None)))
                if count + len(images) > settings.max_dataset_images:
                    raise DatasetError(413, f"数据集图片数超过上限 {settings.max_dataset_images}")
                if not upload_id:
                    session.add(DatasetTask(id=task_id, dataset_id=dataset_id, kind="import", total=len(images)))
                session.flush()
                session.add_all(images)
            committed = True
        except IntegrityError as exc:
            raise DatasetError(409, "数据集已有并发任务，请稍后重试") from exc
        finally:
            if not committed:
                shutil.rmtree(directory)
        logger.info("图片已接管，提交检查任务", extra={"task_id": task_id, "dataset_id": dataset_id, "files": len(images)})
        return self._submit(task_id)

    def schedule(self, dataset_id: str, kind: str, config: AnalysisCreate | VersionCreate, *, task_id: str | None = None) -> DatasetTask:
        """冻结任务输入图片集合，后续导入不会改变该任务的分析范围。

        :param dataset_id: 数据集标识。
        :param kind: 白名单中的 analysis 或 version。
        :param config: 已校验的任务策略。
        :param task_id: 内部工作流的确定性任务 ID；HTTP 调用者不能提供。
        :returns: 已持久化并提交的任务。
        :raises DatasetError: 数据集为空、忙或参数类型不匹配。
        """
        if kind not in {"analysis", "version"} or (kind == "version" and not isinstance(config, VersionCreate)):
            raise DatasetError(422, "不支持的任务类型")
        task_id = task_id or new_id()
        with Session(self.engine) as session:
            existing = session.get(DatasetTask,task_id)
            if existing:
                if existing.dataset_id!=dataset_id or existing.kind!=kind or existing.parameters.get("policy")!=config.model_dump():
                    raise DatasetError(409,"同一数据任务身份不能重放不同输入")
                return self.get_task(task_id)
        try:
            with Session(self.engine) as session, session.begin():
                self._require_idle(session, dataset_id)
                ids = list(session.scalars(select(DatasetImage.id).where(DatasetImage.dataset_id == dataset_id, DatasetImage.deleted_at.is_(None)).order_by(DatasetImage.created_at, DatasetImage.id)))
                if not ids:
                    raise DatasetError(422, "请先导入图片")
                session.add(DatasetTask(id=task_id, dataset_id=dataset_id, kind=kind, total=len(ids), parameters={"policy": config.model_dump(), "image_ids": ids}))
        except IntegrityError as exc:
            raise DatasetError(409, "数据集已有并发任务，请稍后重试") from exc
        return self._submit(task_id)

    def recheck_images(self, dataset_id: str) -> DatasetTask:
        """用新解码规则复查异常原图，禁止改动已经进入冻结版本的资产。"""
        try:
            with Session(self.engine, expire_on_commit=False) as session, session.begin():
                self._require_idle(session, dataset_id)
                ids = list(session.scalars(select(DatasetImage.id).where(
                    DatasetImage.dataset_id == dataset_id, DatasetImage.deleted_at.is_(None), DatasetImage.status.in_(["corrupt", "failed"]),
                    DatasetImage.id.not_in(select(VersionImage.image_id))).order_by(DatasetImage.id)))
                if not ids:
                    raise DatasetError(409, "没有可重新检查的异常原图；已封存版本不会被改写")
                task = DatasetTask(dataset_id=dataset_id, kind="image_recheck", total=len(ids), parameters={"image_ids": ids})
                session.add(task)
            return self._submit(task.id)
        except IntegrityError as error:
            raise DatasetError(409, "数据集已有上传或处理任务") from error

    def get_task(self, task_id: str) -> DatasetTask:
        """读取历史任务，并将 RQ 已终结的异常状态收敛到持久记录。

        :param task_id: 平台任务 ID。
        :returns: 不依赖 Redis 结果 TTL 的持久任务状态。
        :raises DatasetError: 任务不存在。
        """
        with Session(self.engine) as session:
            task = session.get(DatasetTask, task_id)
            if task is None:
                raise DatasetError(404, "数据集任务不存在")
            if task.status in {"queued", "running", "canceling"}:
                reason = None
                queue_origin = None
                try:
                    with redis_connection(self.settings) as connection:
                        job = Job.fetch(task_id, connection=connection, serializer=JSONSerializer)
                        queue_origin = job.origin
                        state = job.get_status(refresh=True)
                        if state in {"failed", "stopped", "canceled"}:
                            reason = f"RQ 任务已终结：{state}，请查看任务日志"
                except NoSuchJobError:
                    age = datetime.now(timezone.utc) - datetime.fromisoformat(task.created_at)
                    if age.total_seconds() > 60:
                        reason = "队列中找不到任务，可能在入队期间进程中断，请重新提交"
                except RedisError:
                    logger.warning("暂时无法读取 RQ 状态，保留数据库任务状态", extra={"task_id": task_id})
                if reason:
                    session.close()
                    fail_dataset_task(self.engine, task_id, reason)
                    from yolo_agent.services.garbage import collect_task_garbage
                    collect_task_garbage(self.engine, self.settings, task_id)
                    return self.get_task(task_id)
                if task.status == "queued" and task.kind == "auto_annotate" and queue_origin:
                    task.progress_detail = {"stage": "annotation_queued", "worker_queue": queue_origin,
                        "independent_queue": queue_origin == self.settings.annotation_queue_name}
            session.expunge(task)
            if task.kind == "cvat_push" and task.status == "running":
                detail = (task.result or {}).get("progress")
                if not detail:
                    # 兼容已经运行的旧 Worker：只读文件计数，不重启、不改写正在进行的任务。
                    run_id = task.parameters.get("run_id")
                    root = owned_path(self.settings, f"annotations/{run_id}/frames")
                    observations = []
                    for path in root.glob("*.png"):
                        try:
                            observations.append(path.stat().st_size)
                        except FileNotFoundError:
                            continue
                    count = min(task.total, len(observations))
                    detail = {"stage": "preparing_images" if count < task.total else "preparing_or_uploading",
                              "completed_images": max(0, count - 1), "total_images": task.total,
                              "prepared_bytes": sum(observations), "estimated": True}
                task.progress_detail = detail
                task.processed = detail.get("completed_images", task.processed)
            elif task.status == "running" and ((task.result or {}).get("progress") or task.kind in {"cvat_pull", "yolo_export"}):
                detail = dict((task.result or {}).get("progress") or {
                    "stage": "reading_review" if task.kind == "cvat_pull" else "export_working",
                    "total_images": task.total, "estimated": True,
                })
                if task.started_at:
                    detail["elapsed_seconds"] = max(0, int((datetime.now(timezone.utc) - datetime.fromisoformat(task.started_at)).total_seconds()))
                export_id = detail.get("export_id")
                if export_id and detail["stage"] in {"export_archiving", "export_verifying_archive"}:
                    archive = owned_path(self.settings, f"exports/{export_id}.zip")
                    if archive.is_file():
                        detail["archive_bytes"] = archive.stat().st_size
                task.progress_detail = detail
            return task

    def manifest_path(self, version_id: str) -> Path:
        """返回经过哈希校验的封存清单，篡改时不提供错误内容。

        :param version_id: 版本标识。
        :returns: 已验证完整性的 JSON 文件路径。
        :raises DatasetError: 版本不存在或清单损坏。
        """
        with Session(self.engine) as session:
            version = session.get(DatasetVersion, version_id)
            if version is None:
                raise DatasetError(404, "数据集版本不存在")
            path = owned_path(self.settings, version.manifest_path)
            if not path.is_file():
                raise DatasetError(409, "版本清单缺失")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != version.manifest_sha256:
                raise DatasetError(409, "版本清单校验失败，拒绝返回被修改的快照")
            return path

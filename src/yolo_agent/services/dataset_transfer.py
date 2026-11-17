"""已标注数据接管与数据快照导出，复用现有任务表及计算队列。"""

import hashlib
import shutil
from pathlib import PurePosixPath

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import AnnotationRun
from yolo_agent.models.dataset import Dataset, DatasetImage, DatasetTask, DatasetVersion, Project, VersionImage, new_id
from yolo_agent.schemas.dataset import LabeledImportOptions
from yolo_agent.services.datasets import DatasetError, DatasetService, owned_path
from yolo_agent.services.inspector import FORMATS
from yolo_agent.services.annotations import assert_reviewed


def submit_dataset_copy(service: DatasetService, source_id: str, data):
    """冻结复制范围，建立目标数据集并复用现有异步数据队列。"""
    identity, target_id = new_id(), new_id()
    try:
        with Session(service.engine) as session, session.begin():
            service._require_idle(session, source_id)
            source = session.get(Dataset, source_id)
            if session.get(Project, data.project_id) is None:
                raise DatasetError(404, "目标项目不存在")
            if source.project_id == data.project_id:
                raise DatasetError(422, "请选择其他项目的数据集")
            if not data.version_id:
                raise DatasetError(422, "只能引用完成标注和审核的版本，请先选择已审版本")
            version = session.get(DatasetVersion, data.version_id)
            if data.version_id and (version is None or version.dataset_id != source_id or not version.sealed):
                raise DatasetError(422, "来源版本必须是所选数据集的已封存版本")
            if version:
                service.manifest_path(version.id)
                ids = list(session.scalars(select(VersionImage.image_id).where(VersionImage.version_id == version.id).order_by(VersionImage.image_id)))
            else:
                ids = list(session.scalars(select(DatasetImage.id).where(DatasetImage.dataset_id == source_id, DatasetImage.deleted_at.is_(None)).order_by(DatasetImage.id)))
            if not ids:
                raise DatasetError(409, "来源数据集没有可复制的图片")
            if len(ids) > service.settings.max_dataset_images:
                raise DatasetError(413, "来源图片数量超过数据集上限")
            run = session.scalar(select(AnnotationRun).where(AnnotationRun.version_id == version.id)
                .order_by(AnnotationRun.created_at.desc(), AnnotationRun.id.desc()).limit(1))
            if not run or run.status not in {"reviewed", "exported"}:
                raise DatasetError(409, "来源版本尚未完成标注和审核，不能引用")
            run_id = run.id
            assert_reviewed(session, run)
            session.add(Dataset(id=target_id, project_id=data.project_id, name=data.name,
                description=f"复制自项目数据集「{source.name}」，后续独立管理。"))
            session.flush()
            session.add(DatasetTask(id=identity, dataset_id=target_id, kind="dataset_copy", total=len(ids),
                parameters={"source_dataset_id": source_id, "source_project_id": source.project_id,
                    "source_dataset_name": source.name, "version_id": data.version_id, "run_id": run_id,
                    "image_ids": ids}))
    except IntegrityError as error:
        raise DatasetError(409, "目标项目已存在同名数据集，或来源正在处理") from error
    return service._submit(identity)


def upload_name(name: str) -> PurePosixPath:
    """上传和 ZIP 共用相对路径边界，不接受目录穿越或 Windows 路径。"""
    path = PurePosixPath(name)
    if (not name or len(name) > 1000 or "\\" in name or ":" in name
            or any(ord(char) < 32 for char in name) or path.is_absolute()
            or ".." in path.parts or path.as_posix() != name):
        raise DatasetError(422, "数据包内文件名必须是安全的相对路径")
    return path


def ignored_file(path: PurePosixPath) -> bool:
    return any(part.startswith(".") or part == "__MACOSX" for part in path.parts) or path.suffix == ".cache"


def submit_labeled_import(service: DatasetService, dataset_id: str, files: list, options: LabeledImportOptions, upload_id: str | None = None):
    """逐块接管浏览器文件；解包、解析、图片检查均由 Worker 执行。"""
    settings = service.settings
    if not files or len(files) > settings.max_dataset_images * 2 + 200:
        raise DatasetError(413, "数据包文件数量超过限制")
    with Session(service.engine) as session:
        service._require_idle(session, dataset_id, excluding=upload_id)
        if upload_id:
            service.require_upload(session, upload_id, dataset_id, "labeled_import")
    identity = upload_id or new_id()
    relative = f"imports/{identity}/upload"
    target = owned_path(settings, relative)
    target.mkdir(parents=True)
    committed, total, seen = False, 0, set()
    archive = len(files) == 1 and PurePosixPath(files[0].filename or "").suffix.lower() == ".zip"
    try:
        for file in files:
            path = upload_name(file.filename or "")
            if ignored_file(path):
                continue
            if path in seen:
                raise DatasetError(422, "数据包包含重复文件名")
            seen.add(path)
            if not archive and path.suffix.lower() not in {*FORMATS, ".txt", ".yaml", ".yml", ".json"}:
                raise DatasetError(422, f"不支持的数据文件：{path.name}")
            destination = target / ("dataset.zip" if archive else path.as_posix())
            destination.parent.mkdir(parents=True, exist_ok=True)
            service.copy_upload(file.file, destination, upload_id, dataset_id, "labeled_import")
            total += destination.stat().st_size
        if not seen:
            raise DatasetError(422, "未找到可导入的数据文件")
        with Session(service.engine) as session, session.begin():
            service._require_idle(session, dataset_id, excluding=upload_id)
            parameters = {**options.model_dump(), "source": relative, "archive": archive, "upload_bytes": total}
            if upload_id:
                changed = session.execute(update(DatasetTask).where(DatasetTask.id == upload_id, DatasetTask.dataset_id == dataset_id, DatasetTask.kind == "labeled_import", DatasetTask.status == "uploading").values(status="queued", parameters=parameters))
                if changed.rowcount != 1:
                    raise DatasetError(409, "上传已取消，文件不会进入导入队列", upload_id)
            else:
                session.add(DatasetTask(id=identity, dataset_id=dataset_id, kind="labeled_import", parameters=parameters))
        committed = True
    except IntegrityError as error:
        raise DatasetError(409, "数据集已有并发任务，请稍后重试") from error
    finally:
        if not committed:
            shutil.rmtree(target.parent)
    return service._submit(identity)


def submit_dataset_export(service: DatasetService, version_id: str):
    """绑定具体版本和最新已完成标注批次，不登记为可训练工件。"""
    identity = new_id()
    try:
        with Session(service.engine) as session, session.begin():
            version = session.get(DatasetVersion, version_id)
            if version is None:
                raise DatasetError(404, "数据集版本不存在")
            if not version.sealed:
                raise DatasetError(409, "数据集版本尚未封存")
            service._require_idle(session, version.dataset_id)
            run_id = session.scalar(select(AnnotationRun.id).where(AnnotationRun.version_id == version_id,
                AnnotationRun.status.in_(["pending_review", "reviewed", "exported"])).order_by(AnnotationRun.created_at.desc(), AnnotationRun.id.desc()).limit(1))
            session.add(DatasetTask(id=identity, dataset_id=version.dataset_id, kind="dataset_export",
                parameters={"version_id": version_id, "run_id": run_id}))
    except IntegrityError as error:
        raise DatasetError(409, "数据集已有并发任务，请稍后重试") from error
    return service._submit(identity)


def download_dataset_export(service: DatasetService, task_id: str):
    task = service.get_task(task_id)
    if task.kind != "dataset_export" or task.status != "completed" or not task.result:
        raise DatasetError(409, "数据集导出尚未完成")
    path = owned_path(service.settings, task.result["archive_path"])
    if not path.is_file():
        raise DatasetError(409, "导出文件缺失，请重新导出")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != task.result["archive_sha256"]:
            raise DatasetError(409, "导出文件完整性校验失败")
    return path

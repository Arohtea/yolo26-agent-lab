"""标注编排和人工审核门禁，不因置信度自动放行。"""

import hashlib
import json
from collections import Counter, defaultdict

from sqlalchemy import Engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import Annotation, AnnotationRun, ClassSchema, DatasetExport, ImageReview, ReviewEvent
from yolo_agent.models.dataset import Dataset, DatasetTask, DatasetVersion, VersionImage, new_id, utc_now
from yolo_agent.schemas.annotation import AnnotationRead, AutoAnnotateCreate, ImageReviewCreate
from yolo_agent.services.datasets import DatasetError, DatasetService, fail_dataset_task
from yolo_agent.settings import Settings
from yolo_agent.workers.queue import enqueue_annotation_task


def annotation_digest(boxes: list[Annotation]) -> str:
    """绑定审核时看到的具体框和状态，避免旧审核凭据被复用。

    :param boxes: 同一图片当前的全部框，包含被拒绝框。
    :returns: 稳定 SHA-256 摘要。
    """
    content = [{"id":box.id,"class_id":box.class_id,"bbox":box.bbox,"status":box.review_status,"flags":box.flags} for box in sorted(boxes, key=lambda box:box.id)]
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_reviewed(session: Session, run: AnnotationRun) -> dict:
    """逐图确认负样本与标注版本，封存之前必须全部审核完成。

    :param session: 当前数据库事务。
    :param run: 已完成推理的标注批次。
    :returns: 类别级人工审核统计。
    :raises DatasetError: 存在待审核、漏标未检查或审核摘要过期。
    """
    if run.status not in {"pending_review", "reviewed", "exported"}:
        raise DatasetError(409, "自动标注尚未完成，不能导出训练数据")
    reviews = list(session.scalars(select(ImageReview).where(ImageReview.run_id == run.id)))
    if len(reviews) != run.total:
        raise DatasetError(409, "图片审核记录不完整")
    all_boxes = list(session.scalars(select(Annotation).where(Annotation.run_id == run.id)))
    by_image = defaultdict(list)
    for box in all_boxes:
        by_image[box.image_id].append(box)
    counts = Counter()
    for review in reviews:
        boxes = by_image[review.image_id]
        if review.status not in {"approved", "corrected"} or not review.missed_objects_checked or not review.reviewer:
            raise DatasetError(409, "所有图片都需人工审核并确认漏标检查，包括没有检测框的图片")
        if any(box.review_status == "pending" for box in boxes) or review.annotation_digest != annotation_digest(boxes):
            raise DatasetError(409, "标注已变化或仍存在待审核框，请重新审核")
        counts.update(box.class_name for box in boxes if box.review_status in {"approved", "corrected"})
    return {"reviewed_images":len(reviews),"class_distribution":dict(counts),"unreviewed_images":0}


class AnnotationService:
    """只调度固定任务，人工审核接口不注册为 Agent 工具。"""

    def __init__(self, engine: Engine, settings: Settings):
        """绑定平台资源。

        :param engine: 当前进程数据库连接池。
        :param settings: 设备、模型与外部服务配置。
        :returns: 无。
        """
        self.engine, self.settings = engine, settings

    def _commit_task(self, task_id: str) -> DatasetTask:
        try:
            enqueue_annotation_task(self.settings, task_id)
        except Exception as exc:
            fail_dataset_task(self.engine, task_id, f"队列提交失败：{type(exc).__name__}")
            with Session(self.engine) as session, session.begin():
                task = session.get(DatasetTask, task_id)
                run = session.get(AnnotationRun, task.parameters["run_id"])
                if task.kind == "auto_annotate":
                    run.status, run.error = "failed", "任务入队失败"
            raise DatasetError(503, "任务入队失败，请检查队列连接", task_id) from exc
        return DatasetService(self.engine, self.settings).get_task(task_id)

    def create_run(self, data: AutoAnnotateCreate, *, run_id: str | None = None, task_id: str | None = None, execution_limits: dict | None = None) -> tuple[AnnotationRun, DatasetTask]:
        """冻结类别和输入版本后创建异步真实检测任务。

        :param data: 类别体系、封存图片版本与阈值。
        :param run_id: 内部工作流的确定性批次 ID。
        :param task_id: 内部工作流的确定性任务 ID。
        :param execution_limits: 编排器收紧的 GPU 内存上限。
        :returns: 待执行批次和对应的 RQ 持久任务。
        """
        if run_id and task_id:
            with Session(self.engine) as session:
                existing=session.get(AnnotationRun,run_id)
                task=session.get(DatasetTask,task_id)
                if existing or task:
                    original = AutoAnnotateCreate.model_validate({key: value for key, value in existing.parameters.items() if key in AutoAnnotateCreate.model_fields}).model_dump() if existing else None
                    if not existing or not task or original!=data.model_dump() or task.parameters.get("run_id")!=run_id:
                        raise DatasetError(409,"标注身份不能重放不同输入")
                    session.expunge(existing)
                    return existing,DatasetService(self.engine,self.settings).get_task(task_id)
        from yolo_agent.services.annotators import validate_source_plan

        # 权重 SHA 校验在写事务之外完成，不让模型文件读取占用 SQLite 写锁。
        with Session(self.engine) as session:
            version = session.get(DatasetVersion, data.version_id)
            schema = session.get(ClassSchema, data.class_schema_id)
            if not version or not schema:
                raise DatasetError(404, "版本或类别体系不存在")
            dataset = session.get(Dataset, version.dataset_id)
            if not version.sealed or schema.project_id != dataset.project_id:
                raise DatasetError(422, "必须使用同项目的类别体系及已封存版本")
            has_test = bool(session.scalar(select(VersionImage.image_id).where(VersionImage.version_id == version.id, VersionImage.split == "test").limit(1)))
            manifest = validate_source_plan(session, self.settings, dataset.project_id, schema.classes, data.class_sources, coco_model=data.coco_model, has_test=has_test)
        versions = {name: entry.get("sha256") or f"{entry['model_id']}@{entry['revision']}:combined-v2" for name, entry in manifest.items()}
        run_id, task_id = run_id or new_id(), task_id or new_id()
        try:
            with Session(self.engine, expire_on_commit=False) as session, session.begin():
                version = session.get(DatasetVersion, data.version_id)
                schema = session.get(ClassSchema, data.class_schema_id)
                if version is None or schema is None:
                    raise DatasetError(404, "版本或类别体系不存在")
                dataset = session.get(Dataset, version.dataset_id)
                if not version.sealed or schema.project_id != dataset.project_id:
                    raise DatasetError(422, "必须使用同项目的类别体系及已封存版本")
                total = session.scalar(select(func.count()).select_from(VersionImage).where(VersionImage.version_id == version.id))
                run = AnnotationRun(id=run_id, version_id=version.id, class_schema_id=schema.id, parameters={**data.model_dump(mode="json"), "_annotators": manifest, "_annotation_revision": 2}, total=total, model_version=json.dumps(versions, sort_keys=True, ensure_ascii=False))
                session.add(run)
                session.add(DatasetTask(id=task_id, dataset_id=dataset.id, kind="auto_annotate", parameters={"run_id":run_id,"execution_limits":execution_limits or {}}, total=total))
        except IntegrityError as exc:
            raise DatasetError(409, "该数据集已有活跃任务，请稍后重试") from exc
        return run, self._commit_task(task_id)

    def schedule(self, run_id: str, operation: str, parameters: dict | None = None, *, task_id: str | None = None) -> DatasetTask:
        """将 CVAT 同步与审核后导出交给 Worker。

        :param run_id: 标注批次。
        :param operation: cvat_push、cvat_pull 或 yolo_export。
        :param parameters: 经过接口校验的人工审核确认信息。
        :param task_id: 内部编排器重放时复用的任务身份。
        :returns: 持久任务记录。
        """
        if operation not in {"cvat_push", "cvat_pull", "yolo_export"}:
            raise DatasetError(422, "不支持的标注操作")
        task_id = task_id or new_id()
        with Session(self.engine) as session:
            existing=session.get(DatasetTask,task_id)
            if existing:
                if existing.kind!=operation or existing.parameters!={"run_id":run_id,**(parameters or {})}:
                    raise DatasetError(409,"标注操作身份不能重放不同输入")
                return DatasetService(self.engine,self.settings).get_task(task_id)
        try:
            with Session(self.engine) as session, session.begin():
                run = session.get(AnnotationRun, run_id)
                if run is None:
                    raise DatasetError(404, "标注批次不存在")
                if run.status not in {"pending_review", "reviewed"}:
                    raise DatasetError(409, "批次尚未完成或已导出冻结")
                version = session.get(DatasetVersion, run.version_id)
                target_schema_id = (parameters or {}).get("reviewed_class_schema_id")
                if operation == "cvat_pull" and target_schema_id:
                    old_schema = session.get(ClassSchema, run.class_schema_id)
                    new_schema = session.get(ClassSchema, target_schema_id)
                    if new_schema is None or new_schema.project_id != old_schema.project_id:
                        raise DatasetError(422, "审核类别版本必须属于同一项目")
                    if new_schema.classes[:len(old_schema.classes)] != old_schema.classes:
                        raise DatasetError(422, "审核同步只允许保留原类别编号并追加新类，重新定义原类请创建新标注批次")
                if operation == "yolo_export":
                    assert_reviewed(session, run)
                if operation in {"cvat_push", "cvat_pull"} and (not self.settings.cvat_username or not self.settings.cvat_password.get_secret_value()):
                    raise DatasetError(503, "请先在 .env 配置本地 CVAT 凭据")
                if operation == "cvat_push" and run.cvat_task_id is not None:
                    raise DatasetError(409, "该批次已关联 CVAT 任务，避免重复上传覆盖审核")
                if operation == "cvat_pull" and run.cvat_task_id is None:
                    raise DatasetError(409, "该批次尚未上传到 CVAT")
                session.add(DatasetTask(id=task_id, dataset_id=version.dataset_id, kind=operation, parameters={"run_id":run_id, **(parameters or {})}, total=run.total))
        except IntegrityError as exc:
            raise DatasetError(409, "数据集已有活跃任务") from exc
        return self._commit_task(task_id)

    def review_image(self, run_id: str, image_id: str, data: ImageReviewCreate) -> dict:
        """人工逐框审核并显式确认漏标检查，拒绝过期审核页面。

        :param run_id: 批次标识。
        :param image_id: 该批次中的图片。
        :param data: 审核者、当前摘要和所有待审框的处理决定。
        :returns: 该图最新审核状态。
        """
        with Session(self.engine) as session, session.begin():
            run = session.get(AnnotationRun, run_id)
            if run is None:
                raise DatasetError(404, "标注批次不存在")
            if run.status not in {"pending_review", "reviewed"}:
                raise DatasetError(409, "当前批次不能审核")
            if run.cvat_task_id:
                raise DatasetError(409, "该批次由 CVAT 管理，请在那里审核后同步")
            version = session.get(DatasetVersion, run.version_id)
            if session.scalar(select(DatasetTask.id).where(DatasetTask.dataset_id == version.dataset_id, DatasetTask.status.in_(["queued", "running"]))):
                raise DatasetError(409, "数据集正在执行任务，暂不能修改审核")
            review = session.get(ImageReview, (run_id, image_id))
            member = session.get(VersionImage, (run.version_id, image_id))
            if review is None or member is None:
                raise DatasetError(404, "图片不属于当前批次")
            boxes = list(session.scalars(select(Annotation).where(Annotation.run_id == run_id, Annotation.image_id == image_id)))
            if annotation_digest(boxes) != data.expected_digest:
                raise DatasetError(409, "标注已变化，请刷新审核内容")
            before = [AnnotationRead.model_validate(box).model_dump() for box in boxes]
            classes = {item["class_id"]:item for item in session.get(ClassSchema, run.class_schema_id).classes}
            lookup = {box.id:box for box in boxes}
            seen = set()
            for decision in data.decisions:
                if decision.class_id not in classes or decision.bbox[2] > member.snapshot["width"] or decision.bbox[3] > member.snapshot["height"]:
                    raise DatasetError(422, "类别不存在或框超出图像边界")
                if decision.annotation_id:
                    box = lookup.get(decision.annotation_id)
                    if box is None or box.id in seen:
                        raise DatasetError(422, "框不属于该图或决定重复")
                    seen.add(box.id)
                    if decision.status == "approved" and (box.bbox != decision.bbox or box.class_id != decision.class_id):
                        raise DatasetError(422, "修改框或类别必须标记为 corrected")
                    box.bbox, box.class_id, box.class_name = decision.bbox, decision.class_id, classes[decision.class_id]["name"]
                    box.review_status, box.reviewer = decision.status, data.reviewer
                else:
                    box = Annotation(id=new_id(), run_id=run_id, image_id=image_id, class_id=decision.class_id, class_name=classes[decision.class_id]["name"], bbox=decision.bbox, confidence=None, annotation_source="human", annotation_model_version="manual", review_status="corrected", reviewer=data.reviewer, flags=[])
                    session.add(box)
                    boxes.append(box)
            if any(box.review_status == "pending" for box in boxes):
                raise DatasetError(409, "仍有待审核框，请逐一接受、修正或拒绝")
            session.flush()
            review.status = "corrected" if any(box.review_status == "corrected" for box in boxes) else "approved"
            review.reviewer, review.missed_objects_checked, review.reviewed_at = data.reviewer, True, utc_now()
            review.annotation_digest = annotation_digest(boxes)
            session.add(ReviewEvent(run_id=run_id, image_id=image_id, reviewer=data.reviewer, source="operator_api", before=before, after=[AnnotationRead.model_validate(box).model_dump() for box in boxes]))
            return {"image_id":image_id,"status":review.status,"annotation_digest":review.annotation_digest}

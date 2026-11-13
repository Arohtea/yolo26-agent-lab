"""标注来源、人工审核证据和可训练数据工件。"""

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from yolo_agent.models.database import Base
from yolo_agent.models.dataset import new_id, utc_now


class ClassSchema(Base):
    """类别体系不可覆盖，变更时创建新的明确版本。"""

    __tablename__ = "class_schemas"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str]
    classes: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(default=utc_now)


class AnnotationRun(Base):
    """自动标注成功只进入待审核状态，不代表数据可训练。"""

    __tablename__ = "annotation_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    class_schema_id: Mapped[str] = mapped_column(ForeignKey("class_schemas.id"))
    status: Mapped[str] = mapped_column(default="queued")
    parameters: Mapped[dict] = mapped_column(JSON)
    model_version: Mapped[str]
    processed: Mapped[int] = mapped_column(default=0)
    total: Mapped[int]
    cvat_task_id: Mapped[int | None]
    cvat_fingerprint: Mapped[str | None]
    error: Mapped[str | None]
    created_at: Mapped[str] = mapped_column(default=utc_now)


class Annotation(Base):
    """框使用 EXIF 校正后的像素坐标，完整保存模型来源和审核状态。"""

    __tablename__ = "annotations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("annotation_runs.id"), index=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), index=True)
    class_id: Mapped[int]
    class_name: Mapped[str]
    bbox: Mapped[list] = mapped_column(JSON)
    confidence: Mapped[float | None]
    annotation_source: Mapped[str]
    annotation_model_version: Mapped[str]
    review_status: Mapped[str] = mapped_column(default="pending")
    flags: Mapped[list] = mapped_column(JSON, default=list)
    reviewer: Mapped[str | None]
    created_at: Mapped[str] = mapped_column(default=utc_now)


class ImageReview(Base):
    """无检测框的图片也必须经过人工漏标检查，防止伪负样本。"""

    __tablename__ = "image_reviews"
    run_id: Mapped[str] = mapped_column(ForeignKey("annotation_runs.id"), primary_key=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), primary_key=True)
    status: Mapped[str] = mapped_column(default="pending")
    missed_objects_checked: Mapped[bool] = mapped_column(default=False)
    reviewer: Mapped[str | None]
    annotation_digest: Mapped[str | None]
    reviewed_at: Mapped[str | None]


class DatasetExport(Base):
    """冻结审核后的 YOLO 数据、划分与工件哈希供实验绑定。"""

    __tablename__ = "dataset_exports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("annotation_runs.id"), index=True)
    directory: Mapped[str]
    manifest_sha256: Mapped[str]
    archive_path: Mapped[str]
    archive_sha256: Mapped[str]
    summary: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(default=utc_now)


class ReviewEvent(Base):
    """保留人工修正前后内容，便于追溯自动标注质量。"""

    __tablename__ = "annotation_review_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("annotation_runs.id"), index=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"))
    reviewer: Mapped[str]
    source: Mapped[str]
    before: Mapped[list] = mapped_column(JSON)
    after: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(default=utc_now)

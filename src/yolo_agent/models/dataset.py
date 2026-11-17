"""数据资产、后台任务与不可变版本的持久化模型。"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, ForeignKey, Index, String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from yolo_agent.models.database import Base


def new_id() -> str:
    """为跨进程任务与数据资产生成无共享计数器的标识。

    :returns: UUID 字符串。
    """
    return str(uuid.uuid4())


def utc_now() -> str:
    """以明确的 UTC 偏移保存 SQLite 时间，避免时区丢失。

    :returns: ISO 8601 时间字符串。
    """
    return datetime.now(timezone.utc).isoformat()


class Project(Base):
    """训练项目，是数据集的业务归属。"""

    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(default="")
    created_at: Mapped[str] = mapped_column(default=utc_now)


class Dataset(Base):
    """可追加图片的数据集，版本快照与其后续追加内容隔离。"""

    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("project_id", "name"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(default="")
    created_at: Mapped[str] = mapped_column(default=utc_now)
    deleted_at: Mapped[str | None] = mapped_column(nullable=True)


class DatasetTask(Base):
    """SQLite 保留任务结果摘要，Redis 过期后仍可追溯。"""

    __tablename__ = "dataset_tasks"
    __table_args__ = (
        Index("uq_dataset_active_task", "dataset_id", unique=True, sqlite_where=text("status IN ('uploading', 'queued', 'running','canceling')")),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    kind: Mapped[str]
    status: Mapped[str] = mapped_column(default="queued")
    stop_requested: Mapped[bool] = mapped_column(default=False)
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None]
    processed: Mapped[int] = mapped_column(default=0)
    total: Mapped[int] = mapped_column(default=0)
    report_path: Mapped[str | None]
    created_at: Mapped[str] = mapped_column(default=utc_now)
    started_at: Mapped[str | None]
    finished_at: Mapped[str | None]


class DatasetImage(Base):
    """原始文件不覆盖，标注前不创建虚构类别或审核状态。"""

    __tablename__ = "images"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    import_task_id: Mapped[str] = mapped_column(ForeignKey("dataset_tasks.id"), index=True)
    original_name: Mapped[str]
    storage_path: Mapped[str]
    status: Mapped[str] = mapped_column(default="pending")
    size_bytes: Mapped[int]
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    phash: Mapped[str | None] = mapped_column(String(16))
    width: Mapped[int | None]
    height: Mapped[int | None]
    format: Mapped[str | None]
    exif_orientation: Mapped[int | None]
    preview_path: Mapped[str | None]
    error: Mapped[str | None]
    video_id: Mapped[str | None]
    scene_id: Mapped[str | None]
    batch_id: Mapped[str | None]
    created_at: Mapped[str] = mapped_column(default=utc_now)
    deleted_at: Mapped[str | None] = mapped_column(nullable=True)


class DatasetVersion(Base):
    """已封存的清单、划分策略与质量分析，不提供更新入口。"""

    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "number"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("dataset_tasks.id"), unique=True)
    number: Mapped[int]
    name: Mapped[str]
    policy: Mapped[dict] = mapped_column(JSON)
    summary: Mapped[dict] = mapped_column(JSON)
    manifest_path: Mapped[str]
    manifest_sha256: Mapped[str]
    sealed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[str] = mapped_column(default=utc_now)


class VersionImage(Base):
    """成员关系和元数据快照均受数据库触发器保护。"""

    __tablename__ = "version_images"
    version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"), primary_key=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), primary_key=True)
    split: Mapped[str] = mapped_column(index=True)
    group_id: Mapped[str]
    snapshot: Mapped[dict] = mapped_column(JSON)

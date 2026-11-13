"""实验配置、训练状态、逐轮指标及独立权重工件。"""

from sqlalchemy import JSON, ForeignKey, Index, String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from yolo_agent.models.database import Base
from yolo_agent.models.dataset import new_id, utc_now


class Experiment(Base):
    """每个实验同时绑定图片版本和人工审核后的数据工件。"""

    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    dataset_version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"))
    dataset_export_id: Mapped[str] = mapped_column(ForeignKey("dataset_exports.id"))
    parent_experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"))
    name: Mapped[str]
    parameters: Mapped[dict] = mapped_column(JSON)
    resolved_parameters: Mapped[dict | None] = mapped_column(JSON)
    optimization_reason: Mapped[str] = mapped_column(default="操作者创建")
    changed_variables: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(default="created")
    final_metrics: Mapped[dict | None] = mapped_column(JSON)
    mlflow_run_id: Mapped[str | None]
    error: Mapped[str | None]
    created_at: Mapped[str] = mapped_column(default=utc_now)


class TrainingTask(Base):
    """第一版同一业务库仅允许一个活跃训练，避免单 GPU 重复占用。"""

    __tablename__ = "training_tasks"
    __table_args__ = (Index("uq_active_training_slot", "resource_slot", unique=True, sqlite_where=text("status IN ('queued','running','canceling')")),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    parent_task_id: Mapped[str | None] = mapped_column(ForeignKey("training_tasks.id"))
    resource_slot: Mapped[str] = mapped_column(default="single-machine")
    status: Mapped[str] = mapped_column(default="queued")
    stop_requested: Mapped[bool] = mapped_column(default=False)
    execution_limits: Mapped[dict] = mapped_column(JSON,default=dict)
    current_epoch: Mapped[int] = mapped_column(default=0)
    total_epochs: Mapped[int]
    device: Mapped[str | None]
    log_path: Mapped[str]
    checkpoint_path: Mapped[str | None]
    checkpoint_sha256: Mapped[str | None]
    best_path: Mapped[str | None]
    mlflow_run_id: Mapped[str | None]
    error: Mapped[str | None]
    elapsed_seconds: Mapped[float] = mapped_column(default=0)
    created_at: Mapped[str] = mapped_column(default=utc_now)
    started_at: Mapped[str | None]
    finished_at: Mapped[str | None]


class ExperimentMetric(Base):
    """指标保存任务与 epoch 来源，恢复训练不会抹去失败尝试记录。"""

    __tablename__ = "experiment_metrics"
    __table_args__ = (UniqueConstraint("task_id", "epoch", "name"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("training_tasks.id"), index=True)
    epoch: Mapped[int]
    name: Mapped[str]
    value: Mapped[float]
    created_at: Mapped[str] = mapped_column(default=utc_now)


class ModelArtifact(Base):
    """注册的是独立复制的权重，不引用还会继续写入的训练检查点。"""

    __tablename__ = "model_registry"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("training_tasks.id"))
    kind: Mapped[str]
    path: Mapped[str]
    sha256: Mapped[str]
    metrics: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(default="candidate")
    created_at: Mapped[str] = mapped_column(default=utc_now)


class ModelExport(Base):
    """部署格式由独立任务转换，原始权重保持只读，同一权重同一格式只保留一份活跃转换。"""

    __tablename__ = "model_exports"
    __table_args__ = (Index("uq_active_model_export", "model_id", "format", unique=True, sqlite_where=text("status IN ('queued','running')")),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model_id: Mapped[str] = mapped_column(ForeignKey("model_registry.id"), index=True)
    format: Mapped[str]
    status: Mapped[str] = mapped_column(default="queued")
    requested_by: Mapped[str] = mapped_column(default="operator")
    reason: Mapped[str] = mapped_column(default="")
    path: Mapped[str | None]
    sha256: Mapped[str | None]
    size_bytes: Mapped[int | None]
    verification: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None]
    created_at: Mapped[str] = mapped_column(default=utc_now)
    started_at: Mapped[str | None]
    finished_at: Mapped[str | None]

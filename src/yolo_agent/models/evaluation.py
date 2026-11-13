"""验证集评估任务与可追溯报告，不读取固定测试集。"""

from sqlalchemy import JSON, ForeignKey, Index, String, text
from sqlalchemy.orm import Mapped, mapped_column
from yolo_agent.models.database import Base
from yolo_agent.models.dataset import new_id, utc_now


class EvaluationTask(Base):
    """模型、数据与评估参数明确绑定，报告以独立文件保存。"""

    __tablename__ = "evaluation_tasks"
    __table_args__ = (Index("uq_active_model_evaluation", "model_id", unique=True, sqlite_where=text("status IN ('queued','running')")),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("model_registry.id"))
    dataset_export_id: Mapped[str] = mapped_column(ForeignKey("dataset_exports.id"))
    status: Mapped[str] = mapped_column(default="queued")
    parameters: Mapped[dict] = mapped_column(JSON)
    execution_limits: Mapped[dict] = mapped_column(JSON,default=dict)
    summary: Mapped[dict | None] = mapped_column(JSON)
    report_path: Mapped[str | None]
    report_sha256: Mapped[str | None]
    mlflow_run_id: Mapped[str | None]
    error: Mapped[str | None]
    created_at: Mapped[str] = mapped_column(default=utc_now)
    started_at: Mapped[str | None]
    finished_at: Mapped[str | None]


class EvaluationInvalidation(Base):
    """发现协议或实现缺陷后另写失效声明，保留原报告及其哈希作为审计。"""

    __tablename__ = "evaluation_invalidations"
    task_id: Mapped[str] = mapped_column(ForeignKey("evaluation_tasks.id"), primary_key=True)
    reason: Mapped[str]
    created_at: Mapped[str] = mapped_column(default=utc_now)

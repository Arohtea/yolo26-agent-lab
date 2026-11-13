"""Agent 的业务状态、不可覆盖事件与实验迭代关联。"""

from sqlalchemy import JSON, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from yolo_agent.models.database import Base
from yolo_agent.models.dataset import new_id, utc_now


class AgentRun(Base):
    """图检查点保存在独立 SQLite，业务库保留可查询状态和操作者预算。"""

    __tablename__ = "agent_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    dataset_version_id: Mapped[str | None] = mapped_column(ForeignKey("dataset_versions.id"))
    class_schema_id: Mapped[str | None] = mapped_column(ForeignKey("class_schemas.id"))
    dataset_export_id: Mapped[str | None] = mapped_column(ForeignKey("dataset_exports.id"))
    name: Mapped[str]
    requirement: Mapped[str] = mapped_column(default="")
    status: Mapped[str] = mapped_column(default="queued")
    phase: Mapped[str] = mapped_column(default="dataset_inspector")
    policy: Mapped[dict] = mapped_column(JSON)
    initial_parameters: Mapped[dict] = mapped_column(JSON)
    state: Mapped[dict] = mapped_column(JSON, default=dict)
    review_confirmation: Mapped[dict | None] = mapped_column(JSON)
    stop_requested: Mapped[bool] = mapped_column(default=False)
    pause_requested: Mapped[bool] = mapped_column(default=False)
    retry_count: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None]
    lease_owner: Mapped[str | None]
    lease_until: Mapped[str | None]
    next_poll_at: Mapped[str | None]
    best_model_id: Mapped[str | None] = mapped_column(ForeignKey("model_registry.id"))
    report_path: Mapped[str | None]
    report_sha256: Mapped[str | None]
    created_at: Mapped[str] = mapped_column(default=utc_now)
    started_at: Mapped[str | None]
    finished_at: Mapped[str | None]


class AgentIteration(Base):
    """同一轮只关联一个实验，重放图节点时不能重复启动计算。"""

    __tablename__ = "agent_iterations"
    __table_args__ = (UniqueConstraint("experiment_id"),)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), primary_key=True)
    number: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"))
    plan: Mapped[dict] = mapped_column(JSON)
    optuna_trial_number: Mapped[int | None]
    evaluation_id: Mapped[str | None] = mapped_column(ForeignKey("evaluation_tasks.id"))
    conclusion: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(default=utc_now)


class AgentEvent(Base):
    """只追加结构化决策及工具结果，不保存凭据和模型内部思考。"""

    __tablename__ = "agent_events"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    node: Mapped[str]
    event_type: Mapped[str]
    level: Mapped[str] = mapped_column(default="info")
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(default=utc_now)

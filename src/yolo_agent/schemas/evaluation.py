"""评估只允许验证集和有界诊断阈值。"""

from pydantic import Field
from yolo_agent.schemas.dataset import InputModel, ORMModel


class EvaluationCreate(InputModel):
    """AP 使用完整低阈值预测，错误清单另用明确的操作阈值。"""

    error_confidence: float = Field(default=0.25, ge=0.01, le=0.99)
    error_iou: float = Field(default=0.5, ge=0.1, le=0.95)
    small_area_pixels: int = Field(default=32*32, ge=1, le=512*512)


class EvaluationRead(ORMModel):
    """报告包含真实数据支持量，无样本类别的 AP 为 null。"""

    id: str
    experiment_id: str
    model_id: str
    dataset_export_id: str
    status: str
    parameters: dict
    summary: dict | None
    report_sha256: str | None
    mlflow_run_id: str | None
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    invalidated_reason: str | None = None

"""训练 Agent 的定向只读查询，限制分页和图片数量。"""

from typing import Literal

from pydantic import Field

from yolo_agent.schemas.dataset import InputModel


class ParameterInfoArgs(InputModel):
    names: list[str] = Field(default_factory=lambda: ["model", "imgsz", "batch", "optimizer", "lr0", "tile", "multi_scale"], min_length=1, max_length=8)
    experiment_id: str | None = None


class ExperimentInspectArgs(InputModel):
    experiment_id: str
    section: Literal["overview", "parameters", "curve", "evaluation", "training"] = "overview"
    offset: int = Field(default=0, ge=0, le=20000)
    limit: int = Field(default=8, ge=1, le=10)
    parameter_names: list[str] = Field(default_factory=list, max_length=8)
    metric_names: list[str] = Field(default_factory=list, max_length=8)
    class_names: list[str] = Field(default_factory=list, max_length=5)


class DatasetExamplesArgs(InputModel):
    dataset_export_id: str | None = None
    split: Literal["train", "val"] = "train"
    class_name: str | None = Field(default=None, max_length=100)
    image_ids: list[str] = Field(default_factory=list, max_length=2)
    offset: int = Field(default=0, ge=0, le=20000)
    limit: int = Field(default=2, ge=1, le=2)
    box_offset: int = Field(default=0, ge=0, le=20000)
    include_images: bool = True


class ValidationErrorsArgs(InputModel):
    experiment_id: str
    class_name: str | None = Field(default=None, max_length=100)
    error_kind: Literal["fn", "fp", "any"] = "fn"
    small_only: bool = Field(default=False, description="只筛选原图小目标漏检；不适用于 fp，不是小目标 AP")
    image_id: str | None = None
    offset: int = Field(default=0, ge=0, le=20000)
    limit: int = Field(default=2, ge=1, le=2)
    box_offset: int = Field(default=0, ge=0, le=20000)
    include_images: bool = True


class CompareExperimentsArgs(InputModel):
    baseline_experiment_id: str
    candidate_experiment_id: str
    class_names: list[str] = Field(default_factory=list, max_length=5)
    offset: int = Field(default=0, ge=0, le=20000)
    limit: int = Field(default=6, ge=1, le=8)
    change_offset: int = Field(default=0, ge=0, le=1000)
    change_limit: int = Field(default=6, ge=1, le=8)


class ConfusionReadArgs(InputModel):
    experiment_id: str
    class_names: list[str] = Field(default_factory=list, max_length=5)
    offset: int = Field(default=0, ge=0, le=20000)
    limit: int = Field(default=5, ge=1, le=5)

"""操作者预算与模型决策使用独立边界，模型不能给自己扩大权限。"""

from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from yolo_agent.schemas.dataset import InputModel, ORMModel
from yolo_agent.schemas.confirmation import MissedObjectsConfirmation
from yolo_agent.schemas.annotation import ClassSources, validate_coco_mapping
from yolo_agent.schemas.experiment import TrainingParameters, training_defaults
from yolo_agent.settings import get_settings


class AgentPolicy(InputModel):
    """一次运行的预算在创建时冻结，计算执行端再次检查。"""

    max_experiments: int = Field(default=20, ge=1, le=20)
    max_training_hours: float = Field(default=48, gt=0, le=48)
    max_retries: int = Field(default=2, ge=0, le=2)
    max_gpu_memory_ratio: float = Field(default=0.9, gt=0.1, le=0.9)
    stagnation_rounds: int = Field(default=8, ge=1, le=20)
    minimum_improvement: float = Field(default=0.001, ge=0, le=0.1)
    allowed_models: list[Literal["yolo26n.pt","yolo26s.pt","yolo26m.pt"]] = Field(default_factory=lambda:["yolo26n.pt","yolo26s.pt"], min_length=1,max_length=3)
    allowed_operations: list[Literal["analyze_dataset","auto_annotate","create_experiment","train","validate","compare","optimize","export_model"]] = Field(default_factory=lambda:["analyze_dataset","auto_annotate","create_experiment","train","validate","compare","optimize","export_model"])
    require_approval: list[Literal["change_class_schema","delete_dataset","publish_model","modify_model_architecture"]] = Field(default_factory=lambda:["change_class_schema","delete_dataset","publish_model","modify_model_architecture"])
    max_epochs_per_experiment: int = Field(default=1000,ge=1,le=1000)
    max_imgsz: int = Field(default=1280,ge=64,le=2560,multiple_of=32)
    max_model_calls: int = Field(default=120,ge=10,le=500)
    purpose: Literal["model_optimization","development_validation"] = "model_optimization"

    @model_validator(mode="after")
    def immutable_approvals(self) -> "AgentPolicy":
        """审批项不可通过一次运行参数移除。

        :returns: 完整权限约束。
        """
        if set(self.require_approval)!={"change_class_schema","delete_dataset","publish_model","modify_model_architecture"}:
            raise ValueError("人工确认项不能减少或重复")
        if len(self.allowed_models)!=len(set(self.allowed_models)) or len(self.require_approval)!=4:
            raise ValueError("权限列表不能重复")
        return self


def agent_defaults() -> AgentPolicy:
    """读取操作者维护的默认策略，模型不能提供配置路径。

    :returns: 已校验的运行策略。
    """
    settings=get_settings()
    path=settings.agent_config_path
    content=yaml.safe_load(path.read_text()) if path.is_file() else {}
    policy=dict(content.get("agent_policy",{}))
    if settings.agent_stagnation_rounds is not None:
        policy["stagnation_rounds"]=settings.agent_stagnation_rounds
    if settings.agent_minimum_improvement is not None:
        policy["minimum_improvement"]=settings.agent_minimum_improvement
    return AgentPolicy.model_validate(policy)


class AgentRunCreate(InputModel):
    """可从原始已导入数据启动，也可复用已审工件跳过重复标注。"""

    dataset_id: str | None = None
    dataset_export_id: str | None = None
    class_schema_id: str | None = None
    name: str = Field(default="自主训练流程",min_length=1,max_length=200)
    requirement: str = Field(default="",max_length=2000)
    annotation_sources: ClassSources = Field(default_factory=dict, max_length=100)
    annotation_box_threshold: float = Field(default=0.35, gt=0, lt=1)
    annotation_coco_model: Literal["yolo26n.pt", "yolo26s.pt"] = "yolo26n.pt"
    policy: AgentPolicy = Field(default_factory=agent_defaults)
    initial_parameters: TrainingParameters = Field(default_factory=training_defaults)

    @field_validator("annotation_sources")
    @classmethod
    def validate_annotation_sources(cls, value):
        return validate_coco_mapping(value)

    @field_validator("policy",mode="before")
    @classmethod
    def inherit_operator_defaults(cls,value):
        """部分请求配置也继承操作者默认值，显式请求字段优先。

        :param value: 原始策略对象或部分 JSON 配置。
        :returns: 已合并的配置，随后仍由 AgentPolicy 校验权限及预算。
        """
        return {**agent_defaults().model_dump(),**value} if isinstance(value,dict) else value

    @model_validator(mode="after")
    def validate_entry(self) -> "AgentRunCreate":
        """入口关系由服务推导，原始数据必须明确使用的类别版本。

        :returns: 已校验请求。
        """
        if bool(self.dataset_id)==bool(self.dataset_export_id):
            raise ValueError("请选择 dataset_id 或已审核 dataset_export_id 中的一个入口")
        if self.dataset_id and not self.class_schema_id:
            raise ValueError("原始数据入口需要现有 class_schema_id，不允许 Agent 创建类别")
        return self


class AgentRunRead(ORMModel):
    """管理界面只读取业务状态，不直接读取或修改图检查点。"""

    id: str
    project_id: str
    dataset_id: str
    dataset_version_id: str | None
    dataset_export_id: str | None
    class_schema_id: str | None
    name: str
    requirement: str
    status: str
    phase: str
    policy: dict
    initial_parameters: dict
    state: dict
    stop_requested: bool
    pause_requested: bool
    retry_count: int
    error: str | None
    best_model_id: str | None
    report_sha256: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class AgentResume(InputModel):
    """人工恢复确认只由管理入口接收，不提供给模型工具。"""

    reviewer: str = Field(min_length=1,max_length=100)
    reason: str = Field(min_length=5,max_length=2000)
    missed_objects_checked: MissedObjectsConfirmation | None = None
    reviewed_class_schema_id: str | None = None


class ClassDraftCreate(InputModel):
    """需求说明只用于生成待人工确认的类别草案，不直接创建类别。"""

    project_id: str
    requirement: str = Field(min_length=4,max_length=2000)


class TrainingSetup(InputModel):
    """人工审核完成后由操作者开始训练：首轮参数可选，预算只能在创建时的上限内收紧。"""

    initial_parameters: dict = Field(default_factory=dict)
    policy: dict = Field(default_factory=dict)

"""训练参数只暴露业务白名单，禁止传入任意 Ultralytics 配置或路径。"""

import math
from typing import Annotated, Literal

import yaml
from pydantic import ConfigDict, Field, model_validator

from yolo_agent.schemas.dataset import InputModel, ORMModel
from yolo_agent.settings import get_settings


OPTIMIZERS = ("auto", "SGD", "MuSGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSprop")
BatchSize = Literal["auto"] | Annotated[int, Field(strict=True, ge=-1, le=256)] | Annotated[float, Field(strict=True, gt=0, lt=1)]
LayerIndex = Annotated[int, Field(strict=True, ge=0, le=20)]
FreezeLayers = LayerIndex | Annotated[list[LayerIndex], Field(max_length=21)]
TileSize = Annotated[bool, Field(strict=True)] | Annotated[int, Field(strict=True, gt=0, le=16384)]


class TrainingParameters(InputModel):
    """模型、显存需求和搜索边界均为有限集合。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    model: Literal["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"] = "yolo26n.pt"
    epochs: int = Field(default=100, ge=1, le=1000)
    imgsz: int = Field(default=640, ge=64, le=2560, multiple_of=32)
    tile: TileSize = Field(default=False, description="高分辨率切片：false 关闭，true 跟随 imgsz，正整数指定原图窗口边长（像素）；训练与独立验证同步使用，增加计算量")
    batch: BatchSize = Field(default="auto", description="auto / -1 自动批量，1–256 为张数，0–1 之间的小数为 CUDA 显存比例；CPU 自动模式使用有界批量")
    optimizer: Literal[OPTIMIZERS] = "auto"
    patience: int = Field(default=20, ge=0, le=200, description="单次训练内部的早停等待 epoch 数，不是 Agent 跨实验停滞轮数")
    pretrained: Literal[True] = True
    seed: int = Field(default=42, ge=0, le=2**32-1, description="随机种子")
    deterministic: bool = Field(default=True, description="尽可能使用确定性运算；跨设备结果仍可能不同")
    lr0: float = Field(default=0.01, gt=0, le=0.1, description="初始学习率；auto 优化器会按模型与训练规模覆盖")
    weight_decay: float = Field(default=0.0005, ge=0, le=0.1, description="权重衰减；实际值还受 batch 与 nbs 缩放")
    mosaic: float = Field(default=1, ge=0, le=1, description="Mosaic 拼图增强概率；另受搜索集合与 rect 模式影响")
    fliplr: float = Field(default=0.5, ge=0, le=1, description="左右翻转概率")

    lrf: float = Field(default=0.01, gt=0, le=1, description="最终学习率相对 lr0 的比例")
    momentum: float = Field(default=0.937, ge=0, lt=1, description="SGD 动量或 Adam 系列的 beta1")
    warmup_epochs: float = Field(default=3.0, ge=0, le=1000, description="预热 epoch 数；允许小数")
    warmup_momentum: float = Field(default=0.8, ge=0, lt=1, description="预热起始动量；Adam 系列 beta1 不参与预热")
    warmup_bias_lr: float = Field(default=0.1, ge=0, le=1, description="预热偏置学习率；auto 优化器会覆盖为 0")
    cos_lr: bool = Field(default=False, description="使用余弦学习率调度")
    nbs: int = Field(default=64, ge=1, le=1024, description="标称 batch，训练库据此计算梯度累积与 weight_decay 缩放")

    box: float = Field(default=7.5, gt=0, le=100, description="边界框损失权重")
    cls: float = Field(default=0.5, gt=0, le=100, description="分类损失权重")
    cls_pw: float = Field(default=0.0, ge=0, le=1, description="按类别频率倒数加权的指数；0 禁用")
    dfl: float = Field(default=1.5, gt=0, le=100, description="距离回归损失权重；YOLO26 无 DFL 时用于 L1 距离损失")

    hsv_h: float = Field(default=0.015, ge=0, le=1, description="HSV 色相扰动幅度")
    hsv_s: float = Field(default=0.7, ge=0, le=1, description="HSV 饱和度扰动幅度")
    hsv_v: float = Field(default=0.4, ge=0, le=1, description="HSV 明度扰动幅度")
    degrees: float = Field(default=0.0, ge=0, le=180, description="随机旋转角度幅度，正负方向对称")
    translate: float = Field(default=0.1, ge=0, le=1, description="随机平移占图像尺寸的比例")
    scale: float = Field(default=0.5, ge=0, le=1, description="随机缩放增益幅度，本平台使用标量形式")
    shear: float = Field(default=0.0, ge=0, le=180, description="随机错切角度幅度，正负方向对称")
    perspective: float = Field(default=0.0, ge=0, le=0.001, description="透视变换幅度；0 禁用")
    flipud: float = Field(default=0.0, ge=0, le=1, description="上下翻转概率，需符合场景物理规律")
    bgr: float = Field(default=0.0, ge=0, le=1, description="RGB 与 BGR 通道交换概率")
    mixup: float = Field(default=0.0, ge=0, le=1, description="两张训练图混合的概率")
    cutmix: float = Field(default=0.0, ge=0, le=1, description="训练图局部区域混合的概率")
    close_mosaic: int = Field(default=10, ge=0, le=1000, description="最后多少个 epoch 关闭 Mosaic 等混合增强；0 不提前关闭")

    multi_scale: float = Field(default=0.0, ge=0, le=1, description="训练尺寸相对 imgsz 的变化幅度；实际最大尺寸仍受运行预算限制")
    rect: bool = Field(default=False, description="矩形训练减少填充；训练库可能关闭不兼容的混合增强")
    freeze: FreezeLayers = Field(default=0, description="冻结前 N 层，或提供层索引列表，如 [0,1,2]；仅开放 0–20，保留检测头训练")

    @model_validator(mode="after")
    def validate_batch(self) -> "TrainingParameters":
        """CPU/GPU 自动批量由执行端解析，显式值必须有界。

        :returns: 已校验参数。
        """
        if isinstance(self.batch, int) and self.batch != -1 and not 1 <= self.batch <= 256:
            raise ValueError("batch 必须为 auto、-1、1–256 整数或 0–1 之间的显存比例")
        if isinstance(self.freeze, list) and len(self.freeze) != len(set(self.freeze)):
            raise ValueError("freeze 层索引不能重复")
        if self.optimizer in {"SGD", "MuSGD"} and self.momentum == 0:
            raise ValueError("SGD / MuSGD 使用 Nesterov，需要 momentum 大于 0")
        if self.multi_scale and int(self.imgsz * (1 - self.multi_scale)) // 32 * 32 < 64:
            raise ValueError("multi_scale 的最小训练尺寸不得小于 64")
        if self.max_training_imgsz() > 2560:
            raise ValueError("多尺度实际最大训练尺寸不得超过 2560")
        return self

    def max_training_imgsz(self) -> int:
        """按训练库的 stride=32 尺寸取整，为 YOLO26 已准备模型计算保守上界。"""
        return math.ceil(self.imgsz * (1 + self.multi_scale) / 32) * 32


# 参数定义、界面字段与 Agent 变化模型均从同一 Pydantic 类型读取，避免白名单漂移。
PARAMETER_GROUPS = {
    "学习率与预热": ("lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "cos_lr", "nbs"),
    "损失权重": ("box", "cls", "cls_pw", "dfl"),
    "颜色增强": ("hsv_h", "hsv_s", "hsv_v", "bgr"),
    "几何增强": ("degrees", "translate", "scale", "shear", "perspective", "fliplr", "flipud"),
    "混合增强": ("mosaic", "mixup", "cutmix", "close_mosaic"),
    "训练策略": ("tile", "multi_scale", "rect", "freeze", "patience", "seed", "deterministic"),
}


def training_parameter_catalog() -> list[dict]:
    """供 Alpine 表单使用的分组字段，校验边界直接来自 Pydantic JSON Schema。"""
    properties = TrainingParameters.model_json_schema()["properties"]
    return [{"name": group, "fields": [{"name": name, **properties[name], "control": "tile" if name == "tile" else "layers" if name == "freeze" else "boolean" if properties[name].get("type") == "boolean" else "number"} for name in names]} for group, names in PARAMETER_GROUPS.items()]


class ExecutionLimits(InputModel):
    """内部编排器只能收紧 Worker 的资源和时间上限。"""

    gpu_memory_ratio: float = Field(default=0.9,gt=0.1,le=0.9)
    timeout_seconds: int = Field(default=172800,ge=1,le=172800)


def training_defaults() -> TrainingParameters:
    """读取操作者管理的默认文件，不能由 Agent 提供文件路径。

    :returns: 已校验的默认训练参数。
    """
    path = get_settings().training_config_path
    return TrainingParameters.model_validate(yaml.safe_load(path.read_text()) if path.is_file() else {})


class ExperimentCreate(InputModel):
    """配置在创建时冻结，下一轮优化必须新建父子关联实验。"""

    dataset_export_id: str
    name: str = Field(default="YOLO26 实验", min_length=1, max_length=200)
    parameters: TrainingParameters = Field(default_factory=training_defaults)
    parent_experiment_id: str | None = None
    optimization_reason: str = Field(default="操作者创建", max_length=4000)


class ExperimentRead(ORMModel):
    """实验状态、配置来源及真实训练指标。"""

    id: str
    project_id: str
    dataset_version_id: str
    dataset_export_id: str
    parent_experiment_id: str | None
    name: str
    parameters: dict
    resolved_parameters: dict | None
    optimization_reason: str
    changed_variables: dict
    status: str
    final_metrics: dict | None
    mlflow_run_id: str | None
    error: str | None
    created_at: str


class TrainingTaskRead(ORMModel):
    """中止请求与真正结束分开呈现。"""

    id: str
    experiment_id: str
    parent_task_id: str | None
    status: str
    stop_requested: bool
    execution_limits: dict
    current_epoch: int
    total_epochs: int
    device: str | None
    mlflow_run_id: str | None
    error: str | None
    elapsed_seconds: float
    created_at: str
    started_at: str | None
    finished_at: str | None


class ModelRead(ORMModel):
    """注册权重不代表发布为生产模型。"""

    id: str
    experiment_id: str
    task_id: str
    kind: str
    sha256: str
    metrics: dict
    status: str
    created_at: str


class ModelExportCreate(InputModel):
    """部署格式为固定白名单，转换参数由执行端从实验配置推导。"""

    format: Literal["onnx", "torchscript"]
    reason: str = Field(default="操作者导出", max_length=2000)


class ModelExportRead(ORMModel):
    """导出文件经过加载与推理校验后才可下载。"""

    id: str
    model_id: str
    format: str
    status: str
    requested_by: str
    reason: str
    sha256: str | None
    size_bytes: int | None
    verification: dict | None
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None

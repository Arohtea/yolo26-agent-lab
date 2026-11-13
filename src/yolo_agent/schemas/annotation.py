"""类别、检测和人工审核输入边界。"""

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, model_validator

from yolo_agent.schemas.dataset import InputModel, ORMModel
from yolo_agent.schemas.confirmation import MissedObjectsConfirmation

# Ultralytics COCO Detection 类别顺序；加载预训练模型时再次与权重 names 核对。
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
CocoClass = Literal[COCO_CLASSES]


class AnnotationSource(InputModel):
    """由操作者按类别确认候选来源，不能传入模型路径或下载地址。"""

    source: Literal["grounding_dino", "yolo26_coco", "project_model"] = "grounding_dino"
    coco_class: CocoClass | None = None
    model_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")

    @model_validator(mode="after")
    def validate_source(self):
        if self.source == "yolo26_coco" and (self.coco_class is None or self.model_id is not None):
            raise ValueError("COCO 来源必须选择 COCO 类别，不能指定项目模型")
        if self.source == "project_model" and (not self.model_id or self.coco_class is not None):
            raise ValueError("项目来源必须选择注册模型；test 图片会回退 DINO")
        if self.source == "grounding_dino" and (self.coco_class is not None or self.model_id is not None):
            raise ValueError("DINO 来源不接受 COCO 映射或项目模型")
        return self


ClassSources = dict[Annotated[int, Field(ge=0, le=999)], AnnotationSource]


def validate_coco_mapping(sources: ClassSources) -> ClassSources:
    """同一个 COCO 类不能被重复映射到两个项目类别。"""
    names = [entry.coco_class for entry in sources.values() if entry.source == "yolo26_coco"]
    if len(names) != len(set(names)):
        raise ValueError("同一 COCO 类不能映射给多个项目类别")
    return sources


class DetectionClass(InputModel):
    """英文提示词用于模型输入，显示名称可以使用中文。"""

    class_id: int = Field(ge=0, le=999)
    name: str = Field(min_length=1, max_length=100)
    prompt: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9 ,'-]{0,79}$")


class ClassSchemaCreate(InputModel):
    """YOLO 类别 ID 必须连续，避免标签与名称错位。"""

    project_id: str
    name: str = Field(min_length=1, max_length=200)
    classes: list[DetectionClass] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_classes(self) -> "ClassSchemaCreate":
        """阻止类别编号、显示名称和提示词歧义。

        :returns: 已校验的类别定义。
        """
        if [item.class_id for item in self.classes] != list(range(len(self.classes))):
            raise ValueError("class_id 必须从 0 开始按顺序连续编号")
        for field in ("name", "prompt"):
            values = [getattr(item, field).lower() for item in self.classes]
            if len(set(values)) != len(values):
                raise ValueError(f"类别 {field} 不得重复")
        return self


class AutoAnnotateCreate(InputModel):
    """只引用已封存版本和明确的类别体系，不接收模型下载地址。"""

    version_id: str
    class_schema_id: str
    class_sources: ClassSources = Field(default_factory=dict, max_length=100)
    coco_model: Literal["yolo26n.pt", "yolo26s.pt"] = "yolo26n.pt"
    box_threshold: float = Field(default=0.35, gt=0, lt=1)
    text_threshold: float = Field(default=0.25, gt=0, lt=1)
    low_confidence_threshold: float = Field(default=0.5, gt=0, lt=1)
    duplicate_iou_threshold: float = Field(default=0.85, gt=0, lt=1)
    max_boxes_per_image: int = Field(default=300, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_sources(self):
        validate_coco_mapping(self.class_sources)
        return self


class BoxDecision(InputModel):
    """人工选择可接受、修正或拒绝框，也能补充漏标框。"""

    annotation_id: str | None = None
    class_id: int = Field(ge=0)
    bbox: list[FiniteFloat] = Field(min_length=4, max_length=4)
    status: Literal["approved", "corrected", "rejected"]

    @model_validator(mode="after")
    def validate_bbox(self) -> "BoxDecision":
        """框必须具有正面积，图像边界在服务层根据真实尺寸检查。

        :returns: 合法的人工审核决定。
        """
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox 必须为有效的 [x1,y1,x2,y2] 像素坐标")
        if self.annotation_id is None and self.status != "corrected":
            raise ValueError("新增漏标框必须标记为 corrected")
        return self


class ImageReviewCreate(InputModel):
    """必须携带当前框摘要，避免审核旧页面覆盖最新结果。"""

    reviewer: str = Field(min_length=1, max_length=100)
    expected_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    missed_objects_checked: MissedObjectsConfirmation
    decisions: list[BoxDecision] = Field(max_length=1000)


class CvatReviewConfirm(InputModel):
    """CVAT 完成人工验收后才能同步为已审核标注。"""

    reviewer: str = Field(min_length=1, max_length=100)
    missed_objects_checked: MissedObjectsConfirmation
    reviewed_class_schema_id: str | None = None


class ClassSchemaRead(ORMModel):
    """不可变类别定义。"""

    id: str
    project_id: str
    name: str
    classes: list[DetectionClass]
    created_at: str


class AnnotationRunRead(ORMModel):
    """批次进度与人工审核入口。"""

    id: str
    version_id: str
    class_schema_id: str
    status: str
    processed: int
    total: int
    model_version: str
    cvat_task_id: int | None
    error: str | None
    created_at: str


class AnnotationRead(ORMModel):
    """每个框的来源、置信度、异常与审核记录。"""

    id: str
    run_id: str
    image_id: str
    class_id: int
    class_name: str
    bbox: list[float]
    confidence: float | None
    annotation_source: str
    annotation_model_version: str
    review_status: str
    flags: list[str]
    reviewer: str | None
    created_at: str


class ExportRead(ORMModel):
    """只暴露工件指纹，不向外泄漏绝对文件路径。"""

    id: str
    version_id: str
    run_id: str
    manifest_sha256: str
    archive_sha256: str
    summary: dict
    created_at: str

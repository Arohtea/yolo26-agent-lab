"""数据导入、检查和划分的明确边界。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GroupField = Literal["video_id", "scene_id", "batch_id"]


class InputModel(BaseModel):
    """拒绝未知参数，避免拼写错误导致安全约束静默失效。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProjectCreate(InputModel):
    """创建训练项目。"""

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)


class DatasetCreate(ProjectCreate):
    """数据集必须绑定已存在的项目。"""

    project_id: str = Field(min_length=36, max_length=36)


class DatasetCopy(InputModel):
    """跨项目建立独立副本；指定版本时保留冻结划分和标注。"""

    project_id: str = Field(min_length=36, max_length=36)
    name: str = Field(min_length=1, max_length=200)
    version_id: str | None = Field(default=None, min_length=36, max_length=36)


class LabeledImportOptions(InputModel):
    """已标注不自动等于已审核；审核继承须由操作者明确声明。"""

    accepted_reviewed_source: bool = False
    reviewer: str = Field(default="", max_length=200)
    confirmation: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def require_attribution(self):
        if self.accepted_reviewed_source and (not self.reviewer or not self.confirmation):
            raise ValueError("继承外部审核需填写审核者及审核来源说明，并确认已经检查漏标")
        return self


class UploadCreate(InputModel):
    """先登记可取消的上传，再接收 multipart；不限制文件字节数。"""

    kind: Literal["import", "labeled_import"] = "import"
    total: int = Field(ge=1)


class AssetDelete(InputModel):
    """操作者核对名称后删除，Agent 工具不暴露此入口。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    confirm_name: str = Field(min_length=1, max_length=1000)


class ImageGroups(InputModel):
    """采集分组由调用者明确提供，不以文件名猜测。"""

    video_id: str | None = Field(default=None, min_length=1, max_length=200)
    scene_id: str | None = Field(default=None, min_length=1, max_length=200)
    batch_id: str | None = Field(default=None, min_length=1, max_length=200)


class AnalysisCreate(InputModel):
    """近似重复仅表示需要审核的候选，不等于标签相同。"""

    phash_threshold: int = Field(default=4, ge=0, le=16)


class VersionCreate(AnalysisCreate):
    """按关联分组冻结三个互斥子集。"""

    name: str = Field(default="数据集快照", min_length=1, max_length=200)
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    val_ratio: float = Field(default=0.1, gt=0, lt=1)
    test_ratio: float = Field(default=0.1, gt=0, lt=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    group_by: list[GroupField] = Field(default_factory=lambda: ["video_id", "scene_id", "batch_id"])

    @model_validator(mode="after")
    def validate_ratios(self) -> "VersionCreate":
        """拒绝无法表达完整数据划分的比例。

        :returns: 已校验的版本配置。
        """
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1) > 1e-8:
            raise ValueError("train_ratio、val_ratio、test_ratio 之和必须为 1")
        if len(self.group_by) != len(set(self.group_by)):
            raise ValueError("group_by 不得重复")
        return self


class ORMModel(BaseModel):
    """从 ORM 读取字段时不会泄露内部存储路径。"""

    model_config = ConfigDict(from_attributes=True)


class ProjectRead(ORMModel):
    """项目查询结果。"""

    id: str
    name: str
    description: str
    created_at: str


class DatasetRead(ProjectRead):
    """数据集基本信息，质量统计由分析任务提供。"""

    project_id: str


class TaskRead(ORMModel):
    """任务状态不依赖短期 RQ 结果保留时间。"""

    id: str
    dataset_id: str
    kind: str
    status: str
    processed: int
    total: int
    result: dict | None
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    progress_detail: dict | None = None


class ImageRead(ORMModel):
    """有效图片可通过固定 ID 的预览接口读取。"""

    id: str
    dataset_id: str
    import_task_id: str
    original_name: str
    status: str
    size_bytes: int
    sha256: str | None
    phash: str | None
    width: int | None
    height: int | None
    format: str | None
    exif_orientation: int | None
    error: str | None
    video_id: str | None
    scene_id: str | None
    batch_id: str | None
    created_at: str


class VersionRead(ORMModel):
    """封存版本摘要；完整成员通过清单及分页接口读取。"""

    id: str
    dataset_id: str
    task_id: str
    number: int
    name: str
    policy: dict
    summary: dict
    manifest_sha256: str
    sealed: bool
    created_at: str


class VersionImageRead(ImageRead):
    """版本清单中的成员保留完整元数据及固定划分。"""

    storage_path: str
    split: Literal["train", "val", "test"]
    group_id: str

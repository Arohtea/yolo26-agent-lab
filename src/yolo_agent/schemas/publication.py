"""人工发布必须确认选中权重指纹和将被替换的生产指针。"""

from typing import Literal
from pydantic import Field, field_validator
from yolo_agent.schemas.dataset import InputModel, ORMModel


class PublishConfirm(InputModel):
    """此输入仅属于人工管理端，不注册为模型工具。"""

    approve: Literal[True]
    reviewer: str=Field(min_length=1,max_length=100)
    reason: str=Field(min_length=5,max_length=2000)
    expected_sha256: str=Field(pattern=r"^[a-f0-9]{64}$")
    expected_current_model_id: str|None=None

    @field_validator("approve",mode="before")
    @classmethod
    def explicit_approval(cls,value):
        """人工发布只接收 JSON true，不能把字符串或数字当作确认。

        :param value: 原始确认字段。
        :returns: 明确的布尔确认。
        """
        if value is not True:
            raise ValueError("发布需要明确的 approve=true")
        return value


class PublicationRead(ORMModel):
    """发布记录不暴露任意路径下载入口。"""

    id: str
    project_id: str
    model_id: str
    previous_model_id: str|None
    evaluation_id: str
    reviewer: str
    reason: str
    weight_sha256: str
    manifest_sha256: str
    created_at: str

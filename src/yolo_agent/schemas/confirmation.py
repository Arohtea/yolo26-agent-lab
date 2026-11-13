"""人工漏标确认必须来自明确布尔输入，不能由类型转换代替。"""

from typing import Annotated, Literal

from pydantic import BeforeValidator


def require_missed_objects_confirmation(value: object) -> bool:
    """在人审契约转换前核对明确确认，避免数字 1 等价于 True。

    :param value: 操作者原始输入。
    :returns: 明确的布尔确认。
    :raises ValueError: 输入不是 JSON true。
    """
    if value is not True:
        raise ValueError("漏标检查需要明确的 missed_objects_checked=true")
    return value


MissedObjectsConfirmation = Annotated[
    Literal[True], BeforeValidator(require_missed_objects_confirmation)
]

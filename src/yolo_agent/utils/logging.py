"""统一结构化日志并限制单个日志文件体积。"""

import logging
from logging.handlers import RotatingFileHandler

from pythonjsonlogger.json import JsonFormatter

from yolo_agent.settings import Settings


def configure_logging(settings: Settings, component: str) -> None:
    """配置终端与按组件分离的滚动日志。

    :param settings: 当前运行配置。
    :param component: 组件名称，用于文件隔离及问题定位。
    :returns: 无。
    """
    settings.prepare_storage()
    formatter = JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(task_id)s %(device)s",
        json_ensure_ascii=False, static_fields={"component": component},
    )
    stream = logging.StreamHandler()
    file = RotatingFileHandler(
        settings.storage_dir / "logs" / f"{component}.log",
        maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    for handler in (stream, file):
        handler.setFormatter(formatter)
    logging.basicConfig(level=settings.log_level, handlers=[stream, file], force=True)

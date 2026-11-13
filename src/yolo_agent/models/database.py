"""SQLite 单机部署连接及短事务基础。"""

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase

from yolo_agent.settings import Settings


class Base(DeclarativeBase):
    """后续业务模型共享的 SQLAlchemy 元数据。"""


def create_database_engine(settings: Settings) -> Engine:
    """为每个进程创建独立连接池并启用 SQLite 并发保护。

    :param settings: 数据路径与锁等待配置。
    :returns: 可供短事务使用的数据库引擎。
    """
    settings.prepare_storage()
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False, "timeout": settings.sqlite_busy_timeout_ms / 1000},
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def configure_connection(connection, _record) -> None:
        """让 API 与独立 Worker 的每条连接遵守相同一致性约束。"""
        cursor = connection.cursor()
        cursor.execute(f"PRAGMA busy_timeout={settings.sqlite_busy_timeout_ms}")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine

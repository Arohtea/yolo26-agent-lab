"""迁移与应用共享配置，避免升级错误的数据库。"""

from alembic import context
import yolo_agent.models.dataset
import yolo_agent.models.annotation
import yolo_agent.models.experiment
import yolo_agent.models.evaluation
import yolo_agent.models.agent
import yolo_agent.models.publication

from yolo_agent.models.database import Base, create_database_engine
from yolo_agent.settings import get_settings


if context.is_offline_mode():
    context.configure(
        url=get_settings().database_url,
        target_metadata=Base.metadata,
        literal_binds=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_database_engine(get_settings())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()

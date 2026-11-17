"""各阶段可终止，停止中的任务继续占用互斥槽。"""

from alembic import op
import sqlalchemy as sa

revision = "0013_task_cancellation"
down_revision = "0012_asset_deletion"
branch_labels = None
depends_on = None

INDEXES = [("dataset_tasks", "uq_dataset_active_task", ["dataset_id"], "'uploading','queued','running','canceling'"),
           ("evaluation_tasks", "uq_active_model_evaluation", ["model_id"], "'queued','running','canceling'"),
           ("model_exports", "uq_active_model_export", ["model_id", "format"], "'queued','running','canceling'")]


def upgrade():
    for table, name, columns, statuses in INDEXES:
        op.add_column(table, sa.Column("stop_requested", sa.Boolean(), nullable=False, server_default=sa.false()))
        op.drop_index(name, table_name=table)
        op.create_index(name, table, columns, unique=True, sqlite_where=sa.text(f"status IN ({statuses})"))


def downgrade():
    for table, name, columns, statuses in reversed(INDEXES):
        op.drop_index(name, table_name=table)
        previous = statuses.replace(",'canceling'", "")
        op.create_index(name, table, columns, unique=True, sqlite_where=sa.text(f"status IN ({previous})"))
        op.drop_column(table, "stop_requested")

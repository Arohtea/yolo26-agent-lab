"""上传接收阶段也占用数据集任务槽，支持事务化取消。"""

from alembic import op
import sqlalchemy as sa

revision = "0011_upload_reservations"
down_revision = "0010_training_setup"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_index("uq_dataset_active_task", table_name="dataset_tasks")
    op.create_index("uq_dataset_active_task", "dataset_tasks", ["dataset_id"], unique=True,
                    sqlite_where=sa.text("status IN ('uploading', 'queued', 'running')"))


def downgrade():
    op.drop_index("uq_dataset_active_task", table_name="dataset_tasks")
    op.create_index("uq_dataset_active_task", "dataset_tasks", ["dataset_id"], unique=True,
                    sqlite_where=sa.text("status IN ('queued', 'running')"))

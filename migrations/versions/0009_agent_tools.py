"""Agent 需求说明与部署格式导出。"""

from alembic import op
import sqlalchemy as sa

revision = '0009_agent_tools'
down_revision = '0008_publication'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """应用当前业务阶段迁移。

    :returns: 无。
    """
    with op.batch_alter_table('agent_runs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('requirement', sa.String(), nullable=False, server_default=''))
    op.create_table('model_exports',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('model_id', sa.String(length=36), nullable=False),
    sa.Column('format', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('requested_by', sa.String(), nullable=False),
    sa.Column('reason', sa.String(), nullable=False),
    sa.Column('path', sa.String(), nullable=True),
    sa.Column('sha256', sa.String(), nullable=True),
    sa.Column('size_bytes', sa.Integer(), nullable=True),
    sa.Column('verification', sa.JSON(), nullable=True),
    sa.Column('error', sa.String(), nullable=True),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.Column('started_at', sa.String(), nullable=True),
    sa.Column('finished_at', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['model_id'], ['model_registry.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('model_exports', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_model_exports_model_id'), ['model_id'], unique=False)
        batch_op.create_index('uq_active_model_export', ['model_id', 'format'], unique=True, sqlite_where=sa.text("status IN ('queued','running')"))


def downgrade() -> None:
    """回滚当前业务阶段迁移。

    :returns: 无。
    """
    with op.batch_alter_table('model_exports', schema=None) as batch_op:
        batch_op.drop_index('uq_active_model_export', sqlite_where=sa.text("status IN ('queued','running')"))
        batch_op.drop_index(batch_op.f('ix_model_exports_model_id'))
    op.drop_table('model_exports')
    with op.batch_alter_table('agent_runs', schema=None) as batch_op:
        batch_op.drop_column('requirement')

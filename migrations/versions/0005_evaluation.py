"""独立验证评估与报告追踪。"""

from alembic import op
import sqlalchemy as sa

revision = '0005_evaluation'
down_revision = '0004_training'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """应用当前业务阶段迁移。

    :returns: 无。
    """
    op.create_table('evaluation_tasks',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('experiment_id', sa.String(length=36), nullable=False),
    sa.Column('model_id', sa.String(length=36), nullable=False),
    sa.Column('dataset_export_id', sa.String(length=36), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('parameters', sa.JSON(), nullable=False),
    sa.Column('summary', sa.JSON(), nullable=True),
    sa.Column('report_path', sa.String(), nullable=True),
    sa.Column('report_sha256', sa.String(), nullable=True),
    sa.Column('mlflow_run_id', sa.String(), nullable=True),
    sa.Column('error', sa.String(), nullable=True),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.Column('started_at', sa.String(), nullable=True),
    sa.Column('finished_at', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_export_id'], ['dataset_exports.id'], ),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
    sa.ForeignKeyConstraint(['model_id'], ['model_registry.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('evaluation_tasks', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_evaluation_tasks_experiment_id'), ['experiment_id'], unique=False)
        batch_op.create_index('uq_active_model_evaluation', ['model_id'], unique=True, sqlite_where=sa.text("status IN ('queued','running')"))


    op.execute("""CREATE TRIGGER frozen_evaluation_binding BEFORE UPDATE OF
        experiment_id,model_id,dataset_export_id,parameters ON evaluation_tasks
        BEGIN SELECT RAISE(ABORT, '评估数据绑定与参数不可变'); END""")
    op.execute("""CREATE TRIGGER frozen_evaluation_result BEFORE UPDATE ON evaluation_tasks
        WHEN OLD.status = 'completed'
        BEGIN SELECT RAISE(ABORT, '完成评估不可覆盖'); END""")


def downgrade() -> None:
    """回滚当前业务阶段迁移。

    :returns: 无。
    """
    op.execute('DROP TRIGGER frozen_evaluation_result')
    op.execute('DROP TRIGGER frozen_evaluation_binding')
    with op.batch_alter_table('evaluation_tasks', schema=None) as batch_op:
        batch_op.drop_index('uq_active_model_evaluation', sqlite_where=sa.text("status IN ('queued','running')"))
        batch_op.drop_index(batch_op.f('ix_evaluation_tasks_experiment_id'))

    op.drop_table('evaluation_tasks')

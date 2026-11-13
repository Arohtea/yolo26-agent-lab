"""实验管理与训练任务"""

from alembic import op
import sqlalchemy as sa

revision = '0004_training'
down_revision = '0003_annotations'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """应用当前业务阶段迁移。

    :returns: 无。
    """
    op.create_table('experiments',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('project_id', sa.String(length=36), nullable=False),
    sa.Column('dataset_version_id', sa.String(length=36), nullable=False),
    sa.Column('dataset_export_id', sa.String(length=36), nullable=False),
    sa.Column('parent_experiment_id', sa.String(length=36), nullable=True),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('parameters', sa.JSON(), nullable=False),
    sa.Column('resolved_parameters', sa.JSON(), nullable=True),
    sa.Column('optimization_reason', sa.String(), nullable=False),
    sa.Column('changed_variables', sa.JSON(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('final_metrics', sa.JSON(), nullable=True),
    sa.Column('mlflow_run_id', sa.String(), nullable=True),
    sa.Column('error', sa.String(), nullable=True),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['dataset_export_id'], ['dataset_exports.id'], ),
    sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_versions.id'], ),
    sa.ForeignKeyConstraint(['parent_experiment_id'], ['experiments.id'], ),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('experiments', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_experiments_project_id'), ['project_id'], unique=False)

    op.create_table('training_tasks',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('experiment_id', sa.String(length=36), nullable=False),
    sa.Column('parent_task_id', sa.String(length=36), nullable=True),
    sa.Column('resource_slot', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('stop_requested', sa.Boolean(), nullable=False),
    sa.Column('current_epoch', sa.Integer(), nullable=False),
    sa.Column('total_epochs', sa.Integer(), nullable=False),
    sa.Column('device', sa.String(), nullable=True),
    sa.Column('log_path', sa.String(), nullable=False),
    sa.Column('checkpoint_path', sa.String(), nullable=True),
    sa.Column('checkpoint_sha256', sa.String(), nullable=True),
    sa.Column('best_path', sa.String(), nullable=True),
    sa.Column('mlflow_run_id', sa.String(), nullable=True),
    sa.Column('error', sa.String(), nullable=True),
    sa.Column('elapsed_seconds', sa.Double(), nullable=False),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.Column('started_at', sa.String(), nullable=True),
    sa.Column('finished_at', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
    sa.ForeignKeyConstraint(['parent_task_id'], ['training_tasks.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('training_tasks', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_training_tasks_experiment_id'), ['experiment_id'], unique=False)
        batch_op.create_index('uq_active_training_slot', ['resource_slot'], unique=True, sqlite_where=sa.text("status IN ('queued','running','canceling')"))

    op.create_table('experiment_metrics',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('experiment_id', sa.String(length=36), nullable=False),
    sa.Column('task_id', sa.String(length=36), nullable=False),
    sa.Column('epoch', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('value', sa.Double(), nullable=False),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
    sa.ForeignKeyConstraint(['task_id'], ['training_tasks.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('task_id', 'epoch', 'name')
    )
    with op.batch_alter_table('experiment_metrics', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_experiment_metrics_experiment_id'), ['experiment_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_experiment_metrics_task_id'), ['task_id'], unique=False)

    op.create_table('model_registry',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('experiment_id', sa.String(length=36), nullable=False),
    sa.Column('task_id', sa.String(length=36), nullable=False),
    sa.Column('kind', sa.String(), nullable=False),
    sa.Column('path', sa.String(), nullable=False),
    sa.Column('sha256', sa.String(), nullable=False),
    sa.Column('metrics', sa.JSON(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
    sa.ForeignKeyConstraint(['task_id'], ['training_tasks.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('model_registry', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_model_registry_experiment_id'), ['experiment_id'], unique=False)

    op.execute("""CREATE TRIGGER frozen_experiment_config BEFORE UPDATE OF
        project_id,dataset_version_id,dataset_export_id,parent_experiment_id,parameters,
        optimization_reason,changed_variables ON experiments
        BEGIN SELECT RAISE(ABORT, '实验来源与配置不可变'); END""")
    op.execute("""CREATE TRIGGER frozen_model_artifact BEFORE UPDATE OF
        experiment_id,task_id,kind,path,sha256,metrics ON model_registry
        BEGIN SELECT RAISE(ABORT, '模型工件不可变'); END""")


def downgrade() -> None:
    """回滚当前业务阶段迁移。

    :returns: 无。
    """
    op.execute('DROP TRIGGER frozen_model_artifact')
    op.execute('DROP TRIGGER frozen_experiment_config')
    with op.batch_alter_table('model_registry', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_model_registry_experiment_id'))

    op.drop_table('model_registry')
    with op.batch_alter_table('experiment_metrics', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_experiment_metrics_task_id'))
        batch_op.drop_index(batch_op.f('ix_experiment_metrics_experiment_id'))

    op.drop_table('experiment_metrics')
    with op.batch_alter_table('training_tasks', schema=None) as batch_op:
        batch_op.drop_index('uq_active_training_slot', sqlite_where=sa.text("status IN ('queued','running','canceling')"))
        batch_op.drop_index(batch_op.f('ix_training_tasks_experiment_id'))

    op.drop_table('training_tasks')
    with op.batch_alter_table('experiments', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_experiments_project_id'))

    op.drop_table('experiments')

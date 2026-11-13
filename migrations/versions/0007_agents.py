"""自主流程、预算、事件与实验关联。"""

from alembic import op
import sqlalchemy as sa

revision = '0007_agents'
down_revision = '0006_evaluation_invalidations'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """应用当前业务阶段迁移。

    :returns: 无。
    """
    op.add_column('training_tasks',sa.Column('execution_limits',sa.JSON(),nullable=False,server_default='{}'))
    op.add_column('evaluation_tasks',sa.Column('execution_limits',sa.JSON(),nullable=False,server_default='{}'))
    op.create_table('agent_runs',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('project_id', sa.String(length=36), nullable=False),
    sa.Column('dataset_id', sa.String(length=36), nullable=False),
    sa.Column('dataset_version_id', sa.String(length=36), nullable=True),
    sa.Column('class_schema_id', sa.String(length=36), nullable=True),
    sa.Column('dataset_export_id', sa.String(length=36), nullable=True),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('phase', sa.String(), nullable=False),
    sa.Column('policy', sa.JSON(), nullable=False),
    sa.Column('initial_parameters', sa.JSON(), nullable=False),
    sa.Column('state', sa.JSON(), nullable=False),
    sa.Column('review_confirmation', sa.JSON(), nullable=True),
    sa.Column('stop_requested', sa.Boolean(), nullable=False),
    sa.Column('pause_requested', sa.Boolean(), nullable=False),
    sa.Column('retry_count', sa.Integer(), nullable=False),
    sa.Column('error', sa.String(), nullable=True),
    sa.Column('lease_owner', sa.String(), nullable=True),
    sa.Column('lease_until', sa.String(), nullable=True),
    sa.Column('next_poll_at', sa.String(), nullable=True),
    sa.Column('best_model_id', sa.String(length=36), nullable=True),
    sa.Column('report_path', sa.String(), nullable=True),
    sa.Column('report_sha256', sa.String(), nullable=True),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.Column('started_at', sa.String(), nullable=True),
    sa.Column('finished_at', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['best_model_id'], ['model_registry.id'], ),
    sa.ForeignKeyConstraint(['class_schema_id'], ['class_schemas.id'], ),
    sa.ForeignKeyConstraint(['dataset_export_id'], ['dataset_exports.id'], ),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_versions.id'], ),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('agent_runs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_agent_runs_project_id'), ['project_id'], unique=False)

    op.create_table('agent_events',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('run_id', sa.String(length=36), nullable=False),
    sa.Column('node', sa.String(), nullable=False),
    sa.Column('event_type', sa.String(), nullable=False),
    sa.Column('level', sa.String(), nullable=False),
    sa.Column('payload', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['agent_runs.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('agent_events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_agent_events_run_id'), ['run_id'], unique=False)

    op.create_table('agent_iterations',
    sa.Column('run_id', sa.String(length=36), nullable=False),
    sa.Column('number', sa.Integer(), nullable=False),
    sa.Column('experiment_id', sa.String(length=36), nullable=False),
    sa.Column('plan', sa.JSON(), nullable=False),
    sa.Column('optuna_trial_number', sa.Integer(), nullable=True),
    sa.Column('evaluation_id', sa.String(length=36), nullable=True),
    sa.Column('conclusion', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['evaluation_id'], ['evaluation_tasks.id'], ),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
    sa.ForeignKeyConstraint(['run_id'], ['agent_runs.id'], ),
    sa.PrimaryKeyConstraint('run_id', 'number'),
    sa.UniqueConstraint('experiment_id')
    )


    op.execute("""CREATE TRIGGER frozen_agent_budget BEFORE UPDATE OF
        project_id,dataset_id,policy,initial_parameters ON agent_runs
        BEGIN SELECT RAISE(ABORT, 'Agent 不能改变运行预算或输入'); END""")
    op.execute("""CREATE TRIGGER frozen_agent_iteration BEFORE UPDATE OF
        run_id,number,experiment_id,plan,optuna_trial_number ON agent_iterations
        BEGIN SELECT RAISE(ABORT, '实验迭代与计划不可覆盖'); END""")
    for operation in ("UPDATE","DELETE"):
        op.execute(f"""CREATE TRIGGER immutable_agent_event_{operation.lower()}
            BEFORE {operation} ON agent_events
            BEGIN SELECT RAISE(ABORT, 'Agent 事件不可覆盖'); END""")


def downgrade() -> None:
    """回滚当前业务阶段迁移。

    :returns: 无。
    """
    op.drop_table('agent_iterations')
    with op.batch_alter_table('agent_events', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_agent_events_run_id'))

    op.drop_table('agent_events')
    with op.batch_alter_table('agent_runs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_agent_runs_project_id'))

    op.drop_table('agent_runs')
    op.drop_column('evaluation_tasks','execution_limits')
    op.drop_column('training_tasks','execution_limits')

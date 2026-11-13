"""运行预算与输入仅在人审后的开始训练门禁处允许操作者收紧。"""

from alembic import op

revision = '0010_training_setup'
down_revision = '0009_agent_tools'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """应用当前业务阶段迁移。

    :returns: 无。
    """
    op.execute("DROP TRIGGER frozen_agent_budget")
    # 项目与数据归属始终不可变；预算和首轮参数只在 training_setup 等待人工时可写，服务层再保证只能收紧。
    op.execute("""CREATE TRIGGER frozen_agent_budget BEFORE UPDATE OF
        project_id,dataset_id,policy,initial_parameters ON agent_runs
        WHEN NEW.project_id IS NOT OLD.project_id OR NEW.dataset_id IS NOT OLD.dataset_id
            OR NOT (OLD.phase = 'training_setup' AND OLD.status = 'needs_review')
        BEGIN SELECT RAISE(ABORT, 'Agent 不能改变运行预算或输入'); END""")


def downgrade() -> None:
    """回滚当前业务阶段迁移。

    :returns: 无。
    """
    op.execute("DROP TRIGGER frozen_agent_budget")
    op.execute("""CREATE TRIGGER frozen_agent_budget BEFORE UPDATE OF
        project_id,dataset_id,policy,initial_parameters ON agent_runs
        BEGIN SELECT RAISE(ABORT, 'Agent 不能改变运行预算或输入'); END""")

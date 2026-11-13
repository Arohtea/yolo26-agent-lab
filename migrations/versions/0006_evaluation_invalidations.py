"""通过追加声明使错误报告失效，不覆盖原始结果。"""

from alembic import op
import sqlalchemy as sa

revision = "0006_evaluation_invalidations"
down_revision = "0005_evaluation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """增加只追加的评估失效审计。

    :returns: 无。
    """
    op.create_table("evaluation_invalidations",
        sa.Column("task_id",sa.String(),sa.ForeignKey("evaluation_tasks.id"),primary_key=True),
        sa.Column("reason",sa.String(),nullable=False),
        sa.Column("created_at",sa.String(),nullable=False))
    for operation in ("UPDATE","DELETE"):
        op.execute(f"""CREATE TRIGGER immutable_evaluation_notice_{operation.lower()}
            BEFORE {operation} ON evaluation_invalidations
            BEGIN SELECT RAISE(ABORT, '评估失效声明不可覆盖'); END""")


def downgrade() -> None:
    """显式回滚审计声明表。

    :returns: 无。
    """
    op.drop_table("evaluation_invalidations")

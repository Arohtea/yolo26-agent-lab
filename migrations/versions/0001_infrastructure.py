"""建立迁移基线，业务表在后续阶段通过独立迁移增加。"""

revision = "0001_infrastructure"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """为数据库建立可追踪的基础版本。

    :returns: 无。
    """
    pass


def downgrade() -> None:
    """移除基础版本标记；当前版本没有业务表。

    :returns: 无。
    """
    pass

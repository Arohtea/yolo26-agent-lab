"""${message}"""

from alembic import op
import sqlalchemy as sa

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    """应用当前业务阶段迁移。

    :returns: 无。
    """
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """回滚当前业务阶段迁移。

    :returns: 无。
    """
    ${downgrades if downgrades else "pass"}

"""人工模型发布审计与生产指针。"""

from alembic import op
import sqlalchemy as sa

revision = '0008_publication'
down_revision = '0007_agents'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """应用当前业务阶段迁移。

    :returns: 无。
    """
    op.create_table('model_publications',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('project_id', sa.String(length=36), nullable=False),
    sa.Column('model_id', sa.String(length=36), nullable=False),
    sa.Column('previous_model_id', sa.String(length=36), nullable=True),
    sa.Column('evaluation_id', sa.String(length=36), nullable=False),
    sa.Column('reviewer', sa.String(), nullable=False),
    sa.Column('reason', sa.String(), nullable=False),
    sa.Column('weight_path', sa.String(), nullable=False),
    sa.Column('weight_sha256', sa.String(), nullable=False),
    sa.Column('manifest_path', sa.String(), nullable=False),
    sa.Column('manifest_sha256', sa.String(), nullable=False),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['evaluation_id'], ['evaluation_tasks.id'], ),
    sa.ForeignKeyConstraint(['model_id'], ['model_registry.id'], ),
    sa.ForeignKeyConstraint(['previous_model_id'], ['model_registry.id'], ),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('model_publications', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_model_publications_project_id'), ['project_id'], unique=False)

    op.create_table('project_production_models',
    sa.Column('project_id', sa.String(length=36), nullable=False),
    sa.Column('model_id', sa.String(length=36), nullable=False),
    sa.Column('publication_id', sa.String(length=36), nullable=False),
    sa.Column('updated_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['model_id'], ['model_registry.id'], ),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.ForeignKeyConstraint(['publication_id'], ['model_publications.id'], ),
    sa.PrimaryKeyConstraint('project_id')
    )

    for operation in ("UPDATE","DELETE"):
        op.execute(f"""CREATE TRIGGER immutable_model_publication_{operation.lower()}
            BEFORE {operation} ON model_publications
            BEGIN SELECT RAISE(ABORT, '人工发布记录不可覆盖'); END""")


def downgrade() -> None:
    """回滚当前业务阶段迁移。

    :returns: 无。
    """
    op.drop_table('project_production_models')
    with op.batch_alter_table('model_publications', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_model_publications_project_id'))

    op.drop_table('model_publications')

"""自动标注与人工审核"""

from alembic import op
import sqlalchemy as sa

revision = '0003_annotations'
down_revision = '0002_datasets'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """应用当前业务阶段迁移。

    :returns: 无。
    """
    op.create_table('class_schemas',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('project_id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('classes', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('class_schemas', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_class_schemas_project_id'), ['project_id'], unique=False)

    op.create_table('annotation_runs',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('version_id', sa.String(length=36), nullable=False),
    sa.Column('class_schema_id', sa.String(length=36), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('parameters', sa.JSON(), nullable=False),
    sa.Column('model_version', sa.String(), nullable=False),
    sa.Column('processed', sa.Integer(), nullable=False),
    sa.Column('total', sa.Integer(), nullable=False),
    sa.Column('cvat_task_id', sa.Integer(), nullable=True),
    sa.Column('cvat_fingerprint', sa.String(), nullable=True),
    sa.Column('error', sa.String(), nullable=True),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['class_schema_id'], ['class_schemas.id'], ),
    sa.ForeignKeyConstraint(['version_id'], ['dataset_versions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('annotation_runs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_annotation_runs_version_id'), ['version_id'], unique=False)

    op.create_table('annotations',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('run_id', sa.String(length=36), nullable=False),
    sa.Column('image_id', sa.String(length=36), nullable=False),
    sa.Column('class_id', sa.Integer(), nullable=False),
    sa.Column('class_name', sa.String(), nullable=False),
    sa.Column('bbox', sa.JSON(), nullable=False),
    sa.Column('confidence', sa.Double(), nullable=True),
    sa.Column('annotation_source', sa.String(), nullable=False),
    sa.Column('annotation_model_version', sa.String(), nullable=False),
    sa.Column('review_status', sa.String(), nullable=False),
    sa.Column('flags', sa.JSON(), nullable=False),
    sa.Column('reviewer', sa.String(), nullable=True),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], ),
    sa.ForeignKeyConstraint(['run_id'], ['annotation_runs.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('annotations', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_annotations_image_id'), ['image_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_annotations_run_id'), ['run_id'], unique=False)

    op.create_table('dataset_exports',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('version_id', sa.String(length=36), nullable=False),
    sa.Column('run_id', sa.String(length=36), nullable=False),
    sa.Column('directory', sa.String(), nullable=False),
    sa.Column('manifest_sha256', sa.String(), nullable=False),
    sa.Column('archive_path', sa.String(), nullable=False),
    sa.Column('archive_sha256', sa.String(), nullable=False),
    sa.Column('summary', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['annotation_runs.id'], ),
    sa.ForeignKeyConstraint(['version_id'], ['dataset_versions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('dataset_exports', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_dataset_exports_run_id'), ['run_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_dataset_exports_version_id'), ['version_id'], unique=False)

    op.create_table('image_reviews',
    sa.Column('run_id', sa.String(length=36), nullable=False),
    sa.Column('image_id', sa.String(length=36), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('missed_objects_checked', sa.Boolean(), nullable=False),
    sa.Column('reviewer', sa.String(), nullable=True),
    sa.Column('annotation_digest', sa.String(), nullable=True),
    sa.Column('reviewed_at', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], ),
    sa.ForeignKeyConstraint(['run_id'], ['annotation_runs.id'], ),
    sa.PrimaryKeyConstraint('run_id', 'image_id')
    )
    op.create_table(
        'annotation_review_events',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('run_id', sa.String(36), sa.ForeignKey('annotation_runs.id'), nullable=False),
        sa.Column('image_id', sa.String(36), sa.ForeignKey('images.id'), nullable=False),
        sa.Column('reviewer', sa.String(), nullable=False),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('before', sa.JSON(), nullable=False),
        sa.Column('after', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.String(), nullable=False),
    )
    op.create_index('ix_annotation_review_events_run_id', 'annotation_review_events', ['run_id'])
    # 类别与审核证据不可覆盖；训练工件一旦冻结，其来源标注同样受到保护。
    for table in ('class_schemas', 'dataset_exports', 'annotation_review_events'):
        for operation in ('UPDATE', 'DELETE'):
            op.execute(f"CREATE TRIGGER immutable_{table}_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable annotation artifact'); END")
    for table in ('annotations', 'image_reviews'):
        for operation in ('INSERT', 'UPDATE', 'DELETE'):
            row = 'NEW' if operation == 'INSERT' else 'OLD'
            op.execute(f"CREATE TRIGGER protect_{table}_{operation.lower()} BEFORE {operation} ON {table} WHEN EXISTS (SELECT 1 FROM dataset_exports WHERE run_id = {row}.run_id) BEGIN SELECT RAISE(ABORT, 'exported annotations are immutable'); END")
    for operation in ('UPDATE', 'DELETE'):
        op.execute(f"CREATE TRIGGER protect_annotation_runs_{operation.lower()} BEFORE {operation} ON annotation_runs WHEN OLD.status = 'exported' BEGIN SELECT RAISE(ABORT, 'exported annotation run is immutable'); END")


def downgrade() -> None:
    """回滚当前业务阶段迁移。

    :returns: 无。
    """
    for table in ('class_schemas', 'dataset_exports', 'annotation_review_events'):
        for operation in ('update', 'delete'):
            op.execute(f"DROP TRIGGER immutable_{table}_{operation}")
    for table in ('annotations', 'image_reviews'):
        for operation in ('insert', 'update', 'delete'):
            op.execute(f"DROP TRIGGER protect_{table}_{operation}")
    for operation in ('update', 'delete'):
        op.execute(f"DROP TRIGGER protect_annotation_runs_{operation}")
    op.drop_table('annotation_review_events')
    op.drop_table('image_reviews')
    with op.batch_alter_table('dataset_exports', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_dataset_exports_version_id'))
        batch_op.drop_index(batch_op.f('ix_dataset_exports_run_id'))

    op.drop_table('dataset_exports')
    with op.batch_alter_table('annotations', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_annotations_run_id'))
        batch_op.drop_index(batch_op.f('ix_annotations_image_id'))

    op.drop_table('annotations')
    with op.batch_alter_table('annotation_runs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_annotation_runs_version_id'))

    op.drop_table('annotation_runs')
    with op.batch_alter_table('class_schemas', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_class_schemas_project_id'))

    op.drop_table('class_schemas')

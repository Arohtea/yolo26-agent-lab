"""数据集管理与版本封存"""

from alembic import op
import sqlalchemy as sa

revision = '0002_datasets'
down_revision = '0001_infrastructure'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """应用当前业务阶段迁移。

    :returns: 无。
    """
    op.create_table('projects',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('description', sa.String(), nullable=False),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('datasets',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('project_id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('description', sa.String(), nullable=False),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('project_id', 'name')
    )
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_datasets_project_id'), ['project_id'], unique=False)

    op.create_table('dataset_tasks',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('dataset_id', sa.String(length=36), nullable=False),
    sa.Column('kind', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('parameters', sa.JSON(), nullable=False),
    sa.Column('result', sa.JSON(), nullable=True),
    sa.Column('error', sa.String(), nullable=True),
    sa.Column('processed', sa.Integer(), nullable=False),
    sa.Column('total', sa.Integer(), nullable=False),
    sa.Column('report_path', sa.String(), nullable=True),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.Column('started_at', sa.String(), nullable=True),
    sa.Column('finished_at', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('dataset_tasks', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_dataset_tasks_dataset_id'), ['dataset_id'], unique=False)
        batch_op.create_index('uq_dataset_active_task', ['dataset_id'], unique=True, sqlite_where=sa.text("status IN ('queued', 'running')"))

    op.create_table('dataset_versions',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('dataset_id', sa.String(length=36), nullable=False),
    sa.Column('task_id', sa.String(length=36), nullable=False),
    sa.Column('number', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('policy', sa.JSON(), nullable=False),
    sa.Column('summary', sa.JSON(), nullable=False),
    sa.Column('manifest_path', sa.String(), nullable=False),
    sa.Column('manifest_sha256', sa.String(), nullable=False),
    sa.Column('sealed', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['task_id'], ['dataset_tasks.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('dataset_id', 'number'),
    sa.UniqueConstraint('task_id')
    )
    with op.batch_alter_table('dataset_versions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_dataset_versions_dataset_id'), ['dataset_id'], unique=False)

    op.create_table('images',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('dataset_id', sa.String(length=36), nullable=False),
    sa.Column('import_task_id', sa.String(length=36), nullable=False),
    sa.Column('original_name', sa.String(), nullable=False),
    sa.Column('storage_path', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('size_bytes', sa.Integer(), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=True),
    sa.Column('phash', sa.String(length=16), nullable=True),
    sa.Column('width', sa.Integer(), nullable=True),
    sa.Column('height', sa.Integer(), nullable=True),
    sa.Column('format', sa.String(), nullable=True),
    sa.Column('exif_orientation', sa.Integer(), nullable=True),
    sa.Column('preview_path', sa.String(), nullable=True),
    sa.Column('error', sa.String(), nullable=True),
    sa.Column('video_id', sa.String(), nullable=True),
    sa.Column('scene_id', sa.String(), nullable=True),
    sa.Column('batch_id', sa.String(), nullable=True),
    sa.Column('created_at', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['import_task_id'], ['dataset_tasks.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('images', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_images_dataset_id'), ['dataset_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_images_import_task_id'), ['import_task_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_images_sha256'), ['sha256'], unique=False)

    op.create_table('version_images',
    sa.Column('version_id', sa.String(length=36), nullable=False),
    sa.Column('image_id', sa.String(length=36), nullable=False),
    sa.Column('split', sa.String(), nullable=False),
    sa.Column('group_id', sa.String(), nullable=False),
    sa.Column('snapshot', sa.JSON(), nullable=False),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], ),
    sa.ForeignKeyConstraint(['version_id'], ['dataset_versions.id'], ),
    sa.PrimaryKeyConstraint('version_id', 'image_id')
    )
    with op.batch_alter_table('version_images', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_version_images_split'), ['split'], unique=False)

    # 封存与成员插入在同一事务完成，后续写入由 SQLite 约束直接拒绝。
    for operation in ('UPDATE', 'DELETE'):
        op.execute(f"CREATE TRIGGER immutable_version_{operation.lower()} BEFORE {operation} ON dataset_versions WHEN OLD.sealed = 1 BEGIN SELECT RAISE(ABORT, 'sealed dataset version is immutable'); END")
        op.execute(f"CREATE TRIGGER immutable_image_{operation.lower()} BEFORE {operation} ON images WHEN EXISTS (SELECT 1 FROM version_images vi JOIN dataset_versions v ON v.id = vi.version_id WHERE vi.image_id = OLD.id AND v.sealed = 1) BEGIN SELECT RAISE(ABORT, 'versioned image is immutable'); END")
    for operation in ('INSERT', 'UPDATE', 'DELETE'):
        if operation == 'INSERT':
            condition = "EXISTS (SELECT 1 FROM dataset_versions WHERE id = NEW.version_id AND sealed = 1)"
        elif operation == 'DELETE':
            condition = "EXISTS (SELECT 1 FROM dataset_versions WHERE id = OLD.version_id AND sealed = 1)"
        else:
            condition = "EXISTS (SELECT 1 FROM dataset_versions WHERE id IN (OLD.version_id, NEW.version_id) AND sealed = 1)"
        op.execute(f"CREATE TRIGGER immutable_membership_{operation.lower()} BEFORE {operation} ON version_images WHEN {condition} BEGIN SELECT RAISE(ABORT, 'sealed version membership is immutable'); END")


def downgrade() -> None:
    """回滚当前业务阶段迁移。

    :returns: 无。
    """
    for operation in ('update', 'delete'):
        op.execute(f"DROP TRIGGER immutable_version_{operation}")
        op.execute(f"DROP TRIGGER immutable_image_{operation}")
    for operation in ('insert', 'update', 'delete'):
        op.execute(f"DROP TRIGGER immutable_membership_{operation}")
    with op.batch_alter_table('version_images', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_version_images_split'))

    op.drop_table('version_images')
    with op.batch_alter_table('images', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_images_sha256'))
        batch_op.drop_index(batch_op.f('ix_images_import_task_id'))
        batch_op.drop_index(batch_op.f('ix_images_dataset_id'))

    op.drop_table('images')
    with op.batch_alter_table('dataset_versions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_dataset_versions_dataset_id'))

    op.drop_table('dataset_versions')
    with op.batch_alter_table('dataset_tasks', schema=None) as batch_op:
        batch_op.drop_index('uq_dataset_active_task', sqlite_where=sa.text("status IN ('queued', 'running')"))
        batch_op.drop_index(batch_op.f('ix_dataset_tasks_dataset_id'))

    op.drop_table('dataset_tasks')
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_datasets_project_id'))

    op.drop_table('datasets')
    op.drop_table('projects')

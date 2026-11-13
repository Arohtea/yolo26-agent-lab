"""原图和数据集可从工作区删除，保留冻结版本的可追溯资产。"""

from alembic import op
import sqlalchemy as sa

revision = "0012_asset_deletion"
down_revision = "0011_upload_reservations"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("datasets", sa.Column("deleted_at", sa.String(), nullable=True))
    op.add_column("images", sa.Column("deleted_at", sa.String(), nullable=True))
    # 工作区可见性不是冻结图像内容；其余原始列仍逐值保护，成员快照完全不改。
    columns = [column["name"] for column in sa.inspect(op.get_bind()).get_columns("images") if column["name"] != "deleted_at"]
    changes = " OR ".join(f'NEW."{name}" IS NOT OLD."{name}"' for name in columns)
    op.execute("DROP TRIGGER immutable_image_update")
    op.execute(f"""CREATE TRIGGER immutable_image_update BEFORE UPDATE ON images
        WHEN EXISTS (SELECT 1 FROM version_images vi JOIN dataset_versions v ON v.id = vi.version_id
                     WHERE vi.image_id = OLD.id AND v.sealed = 1) AND ({changes})
        BEGIN SELECT RAISE(ABORT, 'versioned image is immutable'); END""")


def downgrade():
    op.execute("DROP TRIGGER immutable_image_update")
    op.execute("""CREATE TRIGGER immutable_image_update BEFORE UPDATE ON images
        WHEN EXISTS (SELECT 1 FROM version_images vi JOIN dataset_versions v ON v.id = vi.version_id
                     WHERE vi.image_id = OLD.id AND v.sealed = 1)
        BEGIN SELECT RAISE(ABORT, 'versioned image is immutable'); END""")
    op.drop_column("images", "deleted_at")
    op.drop_column("datasets", "deleted_at")

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "i9d0e1f2g3h4"
down_revision = "h8c9d0e1f2g3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "table_signals",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("record_id", sa.String(256), nullable=False, index=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("col_names", JSONB, nullable=False),
        sa.Column("row_count", sa.Integer, nullable=False),
    )
    op.create_table(
        "table_rows",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("signal_id", sa.Integer, sa.ForeignKey("table_signals.id"), nullable=False, index=True),
        sa.Column("row_idx", sa.Integer, nullable=False),
        sa.Column("col_idx", sa.Integer, nullable=False),
        sa.Column("cell_value", sa.Text, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("table_rows")
    op.drop_table("table_signals")

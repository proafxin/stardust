"""rename doc_id to record_id

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-05-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("tree_nodes", "doc_id", new_column_name="record_id")
    op.alter_column("atoms", "doc_id", new_column_name="record_id")
    op.alter_column("entity_mentions", "doc_id", new_column_name="record_id")
    op.drop_index("ix_tree_nodes_doc_id", table_name="tree_nodes")
    op.drop_index("ix_atoms_doc_id", table_name="atoms")
    op.create_index("ix_tree_nodes_record_id", "tree_nodes", ["record_id"])
    op.create_index("ix_atoms_record_id", "atoms", ["record_id"])


def downgrade() -> None:
    op.alter_column("tree_nodes", "record_id", new_column_name="doc_id")
    op.alter_column("atoms", "record_id", new_column_name="doc_id")
    op.alter_column("entity_mentions", "record_id", new_column_name="doc_id")
    op.drop_index("ix_tree_nodes_record_id", table_name="tree_nodes")
    op.drop_index("ix_atoms_record_id", table_name="atoms")
    op.create_index("ix_tree_nodes_doc_id", "tree_nodes", ["doc_id"])
    op.create_index("ix_atoms_doc_id", "atoms", ["doc_id"])

"""case_updates timestamps use clock_timestamp

``now()`` in PostgreSQL returns the *transaction* start time, so every row
written in a single transaction receives an identical ``created_at``. For an
append-only audit trail that is a defect: two entries appended together cannot
be ordered, and the reporter-visible timeline is exactly such an ordering.

``clock_timestamp()`` reads the wall clock at each statement, so entries always
sort the way they were written. Only ``case_updates`` changes — the other
tables get one row per transaction, where ``now()`` is both correct and the
more stable choice.

Revision ID: 46b59a771f1c
Revises: 63c3addf8378
Create Date: 2026-09-24 16:16:01.861749+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "46b59a771f1c"
down_revision: str | None = "63c3addf8378"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "case_updates",
        "created_at",
        existing_type=postgresql.TIMESTAMP(timezone=True),
        server_default=sa.text("clock_timestamp()"),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "case_updates",
        "created_at",
        existing_type=postgresql.TIMESTAMP(timezone=True),
        server_default=sa.text("now()"),
        existing_nullable=False,
    )

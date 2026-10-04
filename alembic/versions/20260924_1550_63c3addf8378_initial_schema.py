"""initial schema

Creates the entire WhistleDrop AI data layer: four native PostgreSQL enum
types, four tables, their foreign keys, unique indexes and check constraints.

The enum types are created and dropped explicitly rather than implicitly by
``create_table``. ``report_category`` and ``report_status`` are each used by
more than one table, and an implicit create would be attempted once per table,
failing the second time.

Revision ID: 63c3addf8378
Revises:
Create Date: 2026-09-24 15:50:41.114120+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "63c3addf8378"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _enum(name: str, *values: str) -> postgresql.ENUM:
    """A native enum type this migration manages by hand.

    ``create_type=False`` stops ``create_table``/``drop_table`` from trying to
    create or drop the type as a side effect.
    """
    return postgresql.ENUM(*values, name=name, create_type=False)


report_category = _enum(
    "report_category", "SECURITY", "HARASSMENT", "CORRUPTION", "TECHNICAL", "OTHER"
)
report_status = _enum("report_status", "SUBMITTED", "UNDER_REVIEW", "RESOLVED", "DISMISSED")
triage_priority = _enum("triage_priority", "LOW", "MEDIUM", "HIGH", "CRITICAL")
triage_status = _enum("triage_status", "PENDING", "COMPLETED", "FAILED")

ENUM_TYPES = (report_category, report_status, triage_priority, triage_status)


def upgrade() -> None:
    bind = op.get_bind()
    for enum_type in ENUM_TYPES:
        enum_type.create(bind, checkfirst=False)

    # --- moderators --------------------------------------------------------
    op.create_table(
        "moderators",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column(
            "password_hash",
            sa.String(length=255),
            nullable=False,
            comment="Hash only. A plaintext password is never stored or logged.",
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
            comment="Deactivated accounts are retained so their audit history stays attributed.",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_moderators")),
        comment="Staff accounts. The only identified parties in the system.",
    )
    op.create_index(op.f("ix_moderators_username"), "moderators", ["username"], unique=True)

    # --- reports -----------------------------------------------------------
    op.create_table(
        "reports",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "case_code_hash",
            sa.String(length=64),
            nullable=False,
            comment="Keyed hash of the reporter's case code. The plaintext is never stored.",
        ),
        sa.Column(
            "category",
            report_category,
            nullable=False,
            comment="Official category. Set by a human; never by AI triage.",
        ),
        sa.Column(
            "description",
            sa.Text(),
            nullable=False,
            comment="Free-text body of the report. The substance of the record, so never null.",
        ),
        sa.Column(
            "evidence_url",
            sa.Text(),
            nullable=True,
            comment="Optional link to supporting material held elsewhere.",
        ),
        sa.Column(
            "status",
            report_status,
            server_default="SUBMITTED",
            nullable=False,
            comment="Official lifecycle state. Set by a human; never by AI triage.",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reports")),
        comment="Anonymously submitted reports. Contains no reporter identity.",
    )
    op.create_index(op.f("ix_reports_case_code_hash"), "reports", ["case_code_hash"], unique=True)
    op.create_index(
        "ix_reports_status_created_at", "reports", ["status", "created_at"], unique=False
    )

    # --- case_updates ------------------------------------------------------
    op.create_table(
        "case_updates",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("report_id", sa.UUID(), nullable=False),
        sa.Column(
            "from_status",
            report_status,
            nullable=True,
            comment="Status before this entry. Null for the initial submission entry.",
        ),
        sa.Column(
            "to_status",
            report_status,
            nullable=False,
            comment="Status after this entry. Always present.",
        ),
        sa.Column(
            "note",
            sa.Text(),
            nullable=True,
            comment="Optional moderator note explaining the change.",
        ),
        sa.Column(
            "visible_to_reporter",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
            comment="Whether this entry is exposed through case-code lookup.",
        ),
        sa.Column(
            "moderator_id",
            sa.UUID(),
            nullable=True,
            comment="Actor, when there was one. Null for system-generated entries.",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["moderator_id"],
            ["moderators.id"],
            name=op.f("fk_case_updates_moderator_id_moderators"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["report_id"],
            ["reports.id"],
            name=op.f("fk_case_updates_report_id_reports"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_case_updates")),
        comment="Append-only audit trail of report status changes.",
    )
    op.create_index("ix_case_updates_moderator_id", "case_updates", ["moderator_id"], unique=False)
    op.create_index(
        "ix_case_updates_report_id_created_at",
        "case_updates",
        ["report_id", "created_at"],
        unique=False,
    )

    # --- report_triage -----------------------------------------------------
    op.create_table(
        "report_triage",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("report_id", sa.UUID(), nullable=False),
        sa.Column(
            "suggested_category",
            report_category,
            nullable=True,
            comment="Model's guess. Advisory: it never overwrites reports.category.",
        ),
        sa.Column(
            "category_confidence",
            sa.Numeric(precision=4, scale=3),
            nullable=True,
            comment="Model confidence in suggested_category, 0.000-1.000.",
        ),
        sa.Column(
            "suggested_priority",
            triage_priority,
            nullable=True,
            comment="Model's urgency suggestion. Advisory.",
        ),
        sa.Column(
            "keywords",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
            comment="Salient terms extracted from the description.",
        ),
        sa.Column(
            "model_version",
            sa.String(length=64),
            nullable=True,
            comment="Identifier of the model that produced this row, for reproducibility.",
        ),
        sa.Column(
            "status",
            triage_status,
            server_default="PENDING",
            nullable=False,
            comment="State of the inference job. Unrelated to the report's own status.",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(keywords) = 'array'",
            name=op.f("ck_report_triage_keywords_is_array"),
        ),
        sa.CheckConstraint(
            "category_confidence IS NULL"
            " OR (category_confidence >= 0 AND category_confidence <= 1)",
            name=op.f("ck_report_triage_category_confidence_range"),
        ),
        sa.ForeignKeyConstraint(
            ["report_id"],
            ["reports.id"],
            name=op.f("fk_report_triage_report_id_reports"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_report_triage")),
        comment="AI-generated triage suggestions. Advisory only; never authoritative.",
    )
    op.create_index(op.f("ix_report_triage_report_id"), "report_triage", ["report_id"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_report_triage_report_id"), table_name="report_triage")
    op.drop_table("report_triage")

    op.drop_index("ix_case_updates_report_id_created_at", table_name="case_updates")
    op.drop_index("ix_case_updates_moderator_id", table_name="case_updates")
    op.drop_table("case_updates")

    op.drop_index("ix_reports_status_created_at", table_name="reports")
    op.drop_index(op.f("ix_reports_case_code_hash"), table_name="reports")
    op.drop_table("reports")

    op.drop_index(op.f("ix_moderators_username"), table_name="moderators")
    op.drop_table("moderators")

    bind = op.get_bind()
    for enum_type in ENUM_TYPES:
        enum_type.drop(bind, checkfirst=False)

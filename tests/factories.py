"""Builders for isolated test data.

Every builder produces values unique to the call, so two tests can never
collide on ``case_code_hash`` or ``username`` even though they share a schema.
Nothing here writes to the database; the caller decides what to persist.
"""

import uuid

from app.models import CaseUpdate, Moderator, Report, ReportCategory, ReportStatus, ReportTriage


def unique_case_code_hash() -> str:
    """A 64-character hex string, the shape a real case-code hash will have."""
    return uuid.uuid4().hex + uuid.uuid4().hex


def unique_username(prefix: str = "moderator") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def make_report(**overrides: object) -> Report:
    """An unsaved report with plausible, non-identifying content."""
    values: dict[str, object] = {
        "case_code_hash": unique_case_code_hash(),
        "category": ReportCategory.SECURITY,
        "description": "Shared credentials are being reused across production services.",
        "evidence_url": None,
    }
    values.update(overrides)
    return Report(**values)


def make_moderator(**overrides: object) -> Moderator:
    """An unsaved moderator. The hash is a placeholder, not a real one."""
    values: dict[str, object] = {
        "username": unique_username(),
        "password_hash": "not-a-real-hash-phase-3-owns-this",
    }
    values.update(overrides)
    return Moderator(**values)


def make_triage(report: Report, **overrides: object) -> ReportTriage:
    """An unsaved triage row attached to ``report``."""
    values: dict[str, object] = {"report": report}
    values.update(overrides)
    return ReportTriage(**values)


def make_case_update(report: Report, **overrides: object) -> CaseUpdate:
    """An unsaved timeline entry for ``report``."""
    values: dict[str, object] = {
        "report": report,
        "from_status": None,
        "to_status": ReportStatus.SUBMITTED,
    }
    values.update(overrides)
    return CaseUpdate(**values)

"""ORM model registry.

Importing this package imports every model, which is what populates
``Base.metadata``. ``alembic/env.py`` and the test suite both rely on that, so
models must be re-exported here or autogenerate will see an empty schema.
"""

from app.db.base import Base
from app.models.case_update import CaseUpdate
from app.models.enums import (
    ReportCategory,
    ReportStatus,
    TriagePriority,
    TriageStatus,
)
from app.models.moderator import Moderator
from app.models.report import Report
from app.models.triage import ReportTriage

__all__ = [
    "Base",
    "CaseUpdate",
    "Moderator",
    "Report",
    "ReportCategory",
    "ReportStatus",
    "ReportTriage",
    "TriagePriority",
    "TriageStatus",
]

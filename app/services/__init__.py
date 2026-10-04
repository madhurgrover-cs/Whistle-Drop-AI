"""Business logic.

Services sit between routers and repositories. They own transactions and domain
rules, raise :class:`~app.core.errors.AppError` subclasses rather than HTTP
exceptions, and know nothing about requests or responses beyond the schemas they
return.
"""

from app.services.auth import AuthService
from app.services.cases import CaseService
from app.services.moderation import ModerationService
from app.services.reports import ReportService, SubmissionResult

__all__ = [
    "AuthService",
    "CaseService",
    "ModerationService",
    "ReportService",
    "SubmissionResult",
]

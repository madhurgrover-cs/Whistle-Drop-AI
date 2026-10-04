"""Shared FastAPI dependencies.

One place where a request-scoped database session and the application settings
are turned into ready-to-use services. Routers depend on the service, never on
the session directly, which is what keeps database work out of route handlers.

Rate limiting, when it arrives in the hardening phase, slots in here as another
dependency on the route — no change to any service.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AuthenticationError
from app.db.session import get_db
from app.ml.inference import TriageModel
from app.models import Moderator
from app.services.auth import AuthService
from app.services.cases import CaseService
from app.services.moderation import ModerationService
from app.services.reports import ReportService
from app.services.triage import TriageService

DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]


def get_report_service(session: DbSession, settings: AppSettings) -> ReportService:
    """A :class:`ReportService` bound to this request's session."""
    return ReportService(
        session,
        case_code_pepper=settings.case_code_pepper.get_secret_value(),
    )


def get_case_service(session: DbSession, settings: AppSettings) -> CaseService:
    """A :class:`CaseService` bound to this request's session."""
    return CaseService(
        session,
        case_code_pepper=settings.case_code_pepper.get_secret_value(),
    )


def get_auth_service(session: DbSession, settings: AppSettings) -> AuthService:
    """An :class:`AuthService` bound to this request's session."""
    return AuthService(
        session,
        jwt_secret_key=settings.jwt_secret_key.get_secret_value(),
        jwt_algorithm=settings.jwt_algorithm,
        access_token_expire_minutes=settings.jwt_access_token_expire_minutes,
        password_hash_rounds=settings.password_hash_rounds,
    )


@lru_cache(maxsize=4)
def _triage_model(artifact_root: Path, model_version: str) -> TriageModel:
    """One model per (root, version) for the life of the process.

    Cached here rather than inside ``TriageModel`` so that tests can build
    throwaway models freely. Constructing one is cheap — the artifact is not
    read until the first inference — so this is about sharing the *loaded*
    artifact across requests, not about construction cost.
    """
    return TriageModel(artifact_root=artifact_root, model_version=model_version)


def get_triage_service(session: DbSession, settings: AppSettings) -> TriageService:
    """A :class:`TriageService` bound to this request's session.

    Returns a service with no model when triage is switched off, which makes
    the feature flag a single decision here rather than a condition scattered
    through the call path.
    """
    model = (
        _triage_model(settings.ml_artifact_root, settings.ml_model_version)
        if settings.ml_triage_enabled
        else None
    )
    return TriageService(session, model=model)


def get_moderation_service(session: DbSession) -> ModerationService:
    """A :class:`ModerationService` bound to this request's session.

    Takes no secret: moderation reads and writes report state and needs neither
    the case-code pepper nor the token key.
    """
    return ModerationService(session)


ReportServiceDep = Annotated[ReportService, Depends(get_report_service)]
CaseServiceDep = Annotated[CaseService, Depends(get_case_service)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
ModerationServiceDep = Annotated[ModerationService, Depends(get_moderation_service)]
TriageServiceDep = Annotated[TriageService, Depends(get_triage_service)]


# ---------------------------------------------------------------------------
# Moderator authentication
# ---------------------------------------------------------------------------

# auto_error=False so that a missing or malformed header raises this
# application's own AuthenticationError rather than Starlette's default 403
# with its own message. Every authentication failure then leaves through one
# envelope with one status and one body.
_bearer_scheme = HTTPBearer(
    scheme_name="ModeratorBearer",
    description="A moderator access token from `POST /api/v1/auth/login`.",
    auto_error=False,
)

BearerCredentials = Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)]


def get_current_moderator(
    credentials: BearerCredentials,
    service: AuthServiceDep,
) -> Moderator:
    """Resolve the ``Authorization`` header to an active moderator.

    The chain, in order::

        Authorization header -> Bearer scheme -> JWT signature and claims
                             -> moderator lookup -> active check -> Moderator

    Every step fails to the same 401, except an expired token, which is
    reported as such so a client knows to log in again.

    ``HTTPBearer`` enforces the scheme, so ``Authorization: <jwt>`` with no
    ``Bearer``, or ``Basic <...>``, never reaches the token layer at all.
    """
    if credentials is None:
        # No header, or one whose scheme is not Bearer.
        raise AuthenticationError

    if credentials.scheme.lower() != "bearer" or not credentials.credentials.strip():
        raise AuthenticationError

    return service.resolve_moderator(credentials.credentials)


#: Phase 4 endpoints declare ``moderator: CurrentModerator`` to require a login.
CurrentModerator = Annotated[Moderator, Depends(get_current_moderator)]

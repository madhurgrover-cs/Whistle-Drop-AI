"""Moderator authentication.

One endpoint: log in, get a token. There is deliberately **no** registration
route here and there never will be — moderator accounts are created by an
administrator running ``python -m scripts.create_moderator`` against the
database. A whistleblowing system whose moderator set can be joined over HTTP
is not one anybody should file a report into.

Nothing in this module reads the database or verifies a credential; that is all
``app.services.auth``.
"""

from fastapi import APIRouter, Response, status

from app.api.deps import AuthServiceDep
from app.core.errors import ErrorResponse
from app.core.rate_limit import LoginRateLimit
from app.schemas.auth import LoginRequest, TokenResponse

router = APIRouter(tags=["Moderator authentication"])

_DESCRIPTION = """
Exchanges a moderator username and password for a short-lived access token.

### Who can use this

Existing moderators only. There is no sign-up endpoint: accounts are created by
an administrator with database access, using the project's seeding command.

### Using the token

Send it on subsequent requests as:

    Authorization: Bearer <access_token>

It expires after the configured lifetime (30 minutes by default, reported as
`expires_in`). There is no refresh token — when it expires, log in again. An
expired token is answered with `401 TOKEN_EXPIRED`, which is the signal to do
exactly that.

### Failure behaviour

An unknown username, a wrong password and a deactivated account all return the
**same** `401 AUTHENTICATION_FAILED`, with the same body, and take about the
same time to answer. The endpoint will not confirm whether an account exists.

### This does not affect reporting

Filing a report and tracking a case remain fully anonymous and require no
token. Authentication exists only for the people who review reports.

### Rate limiting

Login is the tightest-limited endpoint in the service, because it is the
credential-stuffing surface. Exceeding the limit returns `429
RATE_LIMIT_EXCEEDED` with a `Retry-After` header.
"""

_UNAUTHORISED_EXAMPLE = {
    "error": {"code": "AUTHENTICATION_FAILED", "message": "Authentication failed."}
}


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[LoginRateLimit],
    summary="Log in as a moderator",
    description=_DESCRIPTION,
    responses={
        status.HTTP_200_OK: {
            "description": "Authenticated. The token is in the response body.",
            "content": {
                "application/json": {
                    "example": {
                        "access_token": (
                            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
                            ".eyJzdWIiOiI3YzlkMWIyYS0zZTRmLTQ1NmEtOGI5Yy0xZDJlM2Y0YTViNmMifQ"
                            ".s1gnatur3"
                        ),
                        "token_type": "bearer",
                        "expires_in": 1800,
                    }
                }
            },
        },
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": (
                "Authentication failed. Returned identically for an unknown username, "
                "an incorrect password and a deactivated account."
            ),
            "content": {"application/json": {"example": _UNAUTHORISED_EXAMPLE}},
        },
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ErrorResponse,
            "description": "Too many login attempts. Check the `Retry-After` header.",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": "Too many requests. Please wait a moment and try again.",
                        }
                    }
                }
            },
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "The body was not a valid login request.",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "The request body failed validation.",
                            "details": [{"field": "body.password", "message": "Field required"}],
                        }
                    }
                }
            },
        },
    },
)
def login(
    payload: LoginRequest,
    service: AuthServiceDep,
    response: Response,
) -> TokenResponse:
    # A bearer token must never be written to a shared cache or to the
    # browser's back-forward cache.
    response.headers["Cache-Control"] = "no-store"

    moderator = service.authenticate(
        payload.username,
        payload.password.get_secret_value(),
    )

    # token_type is not passed: TokenResponse defaults it to the RFC 6750
    # constant "bearer", which keeps the literal in one place and out of a
    # call site where a security linter reasonably mistakes it for a password.
    return TokenResponse(
        access_token=service.issue_access_token(moderator),
        expires_in=service.access_token_expires_in_seconds,
    )

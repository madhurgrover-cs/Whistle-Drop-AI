"""Application errors and the single error response envelope.

Every failure the API emits — domain error, validation failure, unhandled
crash — leaves through one of the handlers registered by
:func:`register_exception_handlers` and arrives at the client in the same
shape::

    {"error": {"code": "CASE_NOT_FOUND", "message": "..."}}

Two rules apply throughout, and both exist because this service holds reports
whose contents can endanger the person who filed them:

* **Nothing internal escapes.** No SQL, no stack trace, no DSN, no environment
  value, no exception string from a library. Unhandled exceptions become a flat
  500 with a fixed message.
* **Nothing the reporter submitted is echoed back.** Validation failures report
  *which* field was wrong and *why*, never the value that was rejected —
  otherwise a mistyped case code would be reflected into the response body and
  from there into any proxy or client log.
"""

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.middleware import BASE_SECURITY_HEADERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """One field-level validation problem."""

    field: str = Field(description="Dotted path to the offending field.")
    message: str = Field(description="What is wrong with it.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "field": "body.description",
                    "message": "String should have at least 10 characters",
                }
            ]
        }
    }


class ErrorBody(BaseModel):
    """The contents of the ``error`` key."""

    code: str = Field(description="Stable, machine-readable error identifier.")
    message: str = Field(description="Human-readable explanation. Safe to display.")
    details: list[ErrorDetail] | None = Field(
        default=None,
        description="Field-level problems, present only for validation failures.",
    )


class ErrorResponse(BaseModel):
    """The single error shape returned by every endpoint."""

    error: ErrorBody

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": {
                        "code": "CASE_NOT_FOUND",
                        "message": "No report matches that case code.",
                    }
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


class AppError(Exception):
    """Base class for errors this application raises deliberately.

    Carrying the status code and the public message on the exception keeps
    services free of HTTP imports: a service raises a domain error, and the
    handler below is the only place that knows what an HTTP status is.
    """

    code: str = "INTERNAL_ERROR"
    message: str = "The request could not be completed."
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str | None = None) -> None:
        if message is not None:
            self.message = message
        super().__init__(self.message)


class CaseNotFoundError(AppError):
    """No report matches the supplied case code.

    Raised both for a code that is structurally invalid and for one that is
    well-formed but unknown. See ``app/services/cases.py`` for why those two
    cases are deliberately indistinguishable.
    """

    code = "CASE_NOT_FOUND"
    message = "No report matches that case code."
    status_code = status.HTTP_404_NOT_FOUND


class AuthenticationError(AppError):
    """Authentication failed, for a reason the caller is not told.

    One error class covers every failure on the moderator path: unknown
    username, wrong password, deactivated account, missing header, malformed
    header, bad signature, wrong algorithm, wrong token type. They all produce
    an identical 401 with an identical body.

    That uniformity is the point. A response that distinguished "no such
    moderator" from "wrong password" would turn the login endpoint into an
    account-enumeration oracle, and in a whistleblowing system the set of
    people who moderate reports is itself worth protecting.
    """

    code = "AUTHENTICATION_FAILED"
    message = "Authentication failed."
    status_code = status.HTTP_401_UNAUTHORIZED


class ExpiredCredentialsError(AppError):
    """The access token was well-formed and correctly signed, but has expired.

    The one authentication failure that *is* distinguishable, and safely so: it
    concerns only the bearer's own token and says nothing about any account.
    A client uses it to know that logging in again will help.
    """

    code = "TOKEN_EXPIRED"
    message = "Your session has expired. Please log in again."
    status_code = status.HTTP_401_UNAUTHORIZED


class ReportNotFoundError(AppError):
    """No report exists with the given id.

    Distinct from :class:`CaseNotFoundError`, which answers the *anonymous*
    lookup path and is deliberately vague. This one is raised behind
    authentication, where a moderator asking about a report that does not exist
    learns nothing they could not learn from the queue.
    """

    code = "REPORT_NOT_FOUND"
    message = "No report exists with that id."
    status_code = status.HTTP_404_NOT_FOUND


class InvalidStatusTransitionError(AppError):
    """The requested status change is not a legal move from the current state.

    409 rather than 422: the request is perfectly well formed, and would have
    been accepted had the report been in a different state. The conflict is
    with the resource, not with the body — and a moderator whose colleague has
    already advanced the report needs to see that difference.
    """

    code = "INVALID_STATUS_TRANSITION"
    status_code = status.HTTP_409_CONFLICT

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "That status change is not allowed.")


class RateLimitExceededError(AppError):
    """The caller has made too many requests to this endpoint.

    Carries ``Retry-After`` so a well-behaved client waits rather than
    hammering. The body says only that a limit was reached — not which limit,
    not how many requests remain, and nothing about the caller.
    """

    code = "RATE_LIMIT_EXCEEDED"
    message = "Too many requests. Please wait a moment and try again."
    status_code = status.HTTP_429_TOO_MANY_REQUESTS

    def __init__(self, *, retry_after_seconds: int | None = None) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__()


class RequestTooLargeError(AppError):
    """The request body exceeds the configured maximum."""

    code = "REQUEST_TOO_LARGE"
    message = "The request body is too large."
    status_code = status.HTTP_413_CONTENT_TOO_LARGE


class CaseCodeGenerationError(AppError):
    """Every attempt to mint a unique case code collided.

    Astronomically unlikely by chance; in practice this means something is
    wrong with the random source or the pepper, and failing loudly is correct.
    """

    code = "CASE_CODE_GENERATION_FAILED"
    message = "The report could not be filed. Please try again."
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _envelope(
    status_code: int, body: ErrorBody, headers: dict[str, str] | None = None
) -> JSONResponse:
    """Build an error response, always carrying the hardening headers.

    The headers are applied here as well as in the middleware because the
    handler for an *unhandled* exception runs in Starlette's
    ``ServerErrorMiddleware``, which sits outside everything this application
    installs. Without this, a 500 — the response most worth probing — would be
    the only one served without them. For every other error the middleware
    simply overwrites these with identical values.
    """
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=body).model_dump(exclude_none=True),
        headers={**BASE_SECURITY_HEADERS, **(headers or {})},
    )


def _response_headers(exc: "AppError") -> dict[str, str] | None:
    """Headers a particular domain error needs on its response."""
    if isinstance(exc, RateLimitExceededError) and exc.retry_after_seconds is not None:
        return {"Retry-After": str(exc.retry_after_seconds)}
    return _auth_headers(exc.status_code)


def _auth_headers(status_code: int) -> dict[str, str] | None:
    """A ``WWW-Authenticate`` challenge on 401s, as RFC 9110 requires.

    The realm is all that is advertised. No ``error_description`` is attached:
    that is where implementations usually leak the specific reason a token was
    rejected, which is exactly what this application does not disclose.
    """
    if status_code == status.HTTP_401_UNAUTHORIZED:
        return {"WWW-Authenticate": 'Bearer realm="whistledrop"'}
    return None


async def _handle_app_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    # Logged at the application's own level with no request body attached: the
    # body may hold a case code or the text of a report.
    logger.info("%s on %s %s", exc.code, request.method, request.url.path)
    return _envelope(
        exc.status_code,
        ErrorBody(code=exc.code, message=exc.message),
        _response_headers(exc),
    )


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)

    # Pydantic's raw errors carry an "input" key holding the rejected value, and
    # "ctx" which can hold more of it. Both are dropped: only the location and
    # the reason are safe to return.
    details = [
        ErrorDetail(
            field=".".join(str(part) for part in error.get("loc", ())),
            message=str(error.get("msg", "Invalid value.")),
        )
        for error in exc.errors()
    ]

    return _envelope(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        ErrorBody(
            code="VALIDATION_ERROR",
            message="The request body failed validation.",
            details=details,
        ),
    )


async def _handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    codes = {
        status.HTTP_401_UNAUTHORIZED: "AUTHENTICATION_FAILED",
        status.HTTP_403_FORBIDDEN: "FORBIDDEN",
        status.HTTP_404_NOT_FOUND: "NOT_FOUND",
        status.HTTP_405_METHOD_NOT_ALLOWED: "METHOD_NOT_ALLOWED",
        status.HTTP_413_CONTENT_TOO_LARGE: "REQUEST_TOO_LARGE",
        status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMIT_EXCEEDED",
    }

    # Starlette's own 401 detail ("Not authenticated") would otherwise vary
    # from the application's. One message for every authentication failure.
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        message = AuthenticationError.message
    elif isinstance(exc.detail, str):
        message = exc.detail
    else:
        message = "Request failed."

    return _envelope(
        exc.status_code,
        ErrorBody(code=codes.get(exc.status_code, "HTTP_ERROR"), message=message),
        _auth_headers(exc.status_code),
    )


async def _handle_database_error(request: Request, exc: Exception) -> JSONResponse:
    """Turn any leaked SQLAlchemy error into an opaque 500.

    A DBAPI error message can contain column values, constraint names and
    occasionally fragments of the statement. None of that goes over the wire.
    The full exception is logged server-side, where it belongs.
    """
    logger.exception("Unhandled database error on %s %s", request.method, request.url.path)
    return _envelope(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorBody(code="INTERNAL_ERROR", message="The request could not be completed."),
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _envelope(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorBody(code="INTERNAL_ERROR", message="The request could not be completed."),
    )


Handler = Callable[[Request, Exception], Awaitable[JSONResponse]]


def register_exception_handlers(app: FastAPI) -> None:
    """Wire every handler onto ``app``. Called once from the application factory."""
    handlers: list[tuple[type[Exception] | int, Handler]] = [
        (AppError, _handle_app_error),
        (RequestValidationError, _handle_validation_error),
        (StarletteHTTPException, _handle_http_exception),
        (SQLAlchemyError, _handle_database_error),
        (Exception, _handle_unexpected_error),
    ]
    for exception_type, handler in handlers:
        app.add_exception_handler(exception_type, handler)

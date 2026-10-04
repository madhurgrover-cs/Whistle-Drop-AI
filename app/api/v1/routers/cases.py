"""Case tracking.

Why this is a POST with a body, and not ``GET /cases/{case_code}``
------------------------------------------------------------------

A case code is a **bearer credential**: possession of it is the entire
authorisation to read a report. Anything placed in a URL path or query string
leaks by default, into places nobody audits and everybody keeps:

* browser history and the address bar, on whatever machine the reporter used —
  often a shared or work machine;
* the ``Referer`` header sent to any third party the page later links to;
* web server and reverse-proxy access logs, which almost always record the full
  request line, are retained for months, and are shipped to log aggregators that
  a far wider group of staff can read than can read the reports database;
* CDN, WAF and load-balancer telemetry;
* bookmarks, "copy link" and shoulder-surfing.

A request body appears in none of those. It is not a confidentiality guarantee
on its own — TLS is what protects it in transit — but it removes an entire class
of accidental disclosure for free. The cost is one broken REST convention: this
is a read, expressed as a POST. That trade is the right way round when the
alternative is writing a credential into every log on the path.

The endpoint is therefore also declared non-cacheable, for the same reason.
"""

from fastapi import APIRouter, Response, status

from app.api.deps import CaseServiceDep
from app.core.errors import ErrorResponse
from app.core.rate_limit import CaseLookupRateLimit
from app.schemas.cases import CaseLookupRequest, CaseLookupResponse

router = APIRouter(tags=["Case tracking"])

_DESCRIPTION = """
Looks up a report using the case code issued when it was filed.

### No account needed

Authentication is neither required nor possible. The case code *is* the
credential — anyone holding it can read this response, and anyone without it
cannot. Treat it like a password.

### What you get back

The report's current status, the category it is filed under, when it was
submitted and last changed, and the history a moderator has chosen to publish.

### What you never get back

Internal database identifiers, the stored hash of your case code, the identity
of any moderator, internal notes, and anything from the AI triage layer. Updates
marked internal are filtered out in the database query itself, so they are never
loaded into the response at all.

### Unknown codes

A code that does not exist and a code that is malformed produce the **same**
404. The endpoint will not confirm that a guess had the right shape.

### Lost codes

A lost case code cannot be recovered. Only a keyed hash of it is stored, and any
recovery route would require knowing who filed the report — which is exactly
what this system refuses to record.

### Rate limiting

This endpoint is rate limited per client address. Exceeding it returns `429
RATE_LIMIT_EXCEEDED` with a `Retry-After` header. The limit is abuse control
only — it is not an anonymity mechanism, and the address it counts against is
hashed with a per-process salt, never stored and never logged.
"""


@router.post(
    "/cases/lookup",
    response_model=CaseLookupResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[CaseLookupRateLimit],
    summary="Track a report using its case code",
    description=_DESCRIPTION,
    responses={
        status.HTTP_200_OK: {
            "description": "The reporter-visible view of the report.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "UNDER_REVIEW",
                        "category": "SECURITY",
                        "submitted_at": "2026-09-24T16:10:31.482000Z",
                        "last_updated_at": "2026-09-25T09:02:14.907000Z",
                        "timeline": [
                            {
                                "status": "SUBMITTED",
                                "note": "Report received. It is queued for review by a moderator.",
                                "occurred_at": "2026-09-24T16:10:31.482000Z",
                            },
                            {
                                "status": "UNDER_REVIEW",
                                "note": "A moderator has begun reviewing this report.",
                                "occurred_at": "2026-09-25T09:02:14.907000Z",
                            },
                        ],
                    }
                }
            },
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "No report matches that code — or the code was malformed. "
            "The two responses are identical on purpose.",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "CASE_NOT_FOUND",
                            "message": "No report matches that case code.",
                        }
                    }
                }
            },
        },
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ErrorResponse,
            "description": "Rate limit exceeded. Check the `Retry-After` header.",
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
            "description": "The body was not a valid lookup request.",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "The request body failed validation.",
                            "details": [{"field": "body.case_code", "message": "Field required"}],
                        }
                    }
                }
            },
        },
    },
)
def lookup_case(
    payload: CaseLookupRequest,
    service: CaseServiceDep,
    response: Response,
) -> CaseLookupResponse:
    # The body holds a credential and the reply holds report contents. Neither
    # belongs in a shared cache or a browser's back-forward cache.
    response.headers["Cache-Control"] = "no-store"
    return service.lookup(payload.case_code)

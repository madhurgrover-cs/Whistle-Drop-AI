"""Anonymous report submission.

The handler is four lines of work: hand the validated body to the service, hand
the result back as a response model. Case-code generation, hashing, the
timeline entry, the transaction and the collision retry all live in
``app.services.reports``.
"""

from fastapi import APIRouter, status

from app.api.deps import ReportServiceDep, TriageServiceDep
from app.core.errors import ErrorResponse
from app.core.rate_limit import ReportSubmissionRateLimit
from app.schemas.reports import ReportSubmissionRequest, ReportSubmissionResponse

router = APIRouter(tags=["Reports"])

_DESCRIPTION = """
Files a report anonymously and returns a **case code**.

### No account, no identity

This endpoint requires no authentication and accepts no identifying field. There
is no name, email, phone or account parameter, and sending an unrecognised field
is rejected rather than ignored. The database has no reporters table for such a
value to go into.

### The case code is shown exactly once

The response contains the only plaintext copy of your case code that will ever
exist. Only a keyed hash of it is stored, so **nobody can recover or reissue
it** — not a moderator, not an administrator, not someone holding a full
database dump. Save it before you close the page.

### What this does and does not protect

Your identity is not collected, stored or inferred by this service. That is an
*application-level* guarantee, and it is not the same as anonymity on the
network: your IP address still reaches whatever terminates TLS in front of this
API, and submission timing is still observable. Use Tor or a VPN if your threat
model requires more. A report can also identify its author through its own
contents — write accordingly.
"""


@router.post(
    "/reports",
    response_model=ReportSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
    # Rate limiting is a route concern: the dependency runs before the handler
    # and raises, so the service below never learns it exists.
    dependencies=[ReportSubmissionRateLimit],
    summary="Submit an anonymous report",
    description=_DESCRIPTION,
    responses={
        status.HTTP_201_CREATED: {
            "description": "Report filed. Save the case code — it is not shown again.",
            "content": {
                "application/json": {
                    "example": {
                        "case_code": "WD-4K7PQ-92MRT-XJ3HN-B8VZ6",
                        "status": "SUBMITTED",
                        "submitted_at": "2026-09-24T16:10:31.482000Z",
                        "message": (
                            "Save this case code somewhere safe. It is the only way to "
                            "check your report, and it cannot be recovered or reissued "
                            "if you lose it."
                        ),
                    }
                }
            },
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "The body failed validation.",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "The request body failed validation.",
                            "details": [
                                {
                                    "field": "body.description",
                                    "message": "String should have at least 10 characters",
                                }
                            ],
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
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "A unique case code could not be minted. Retry the request.",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "CASE_CODE_GENERATION_FAILED",
                            "message": "The report could not be filed. Please try again.",
                        }
                    }
                }
            },
        },
    },
)
def submit_report(
    payload: ReportSubmissionRequest,
    service: ReportServiceDep,
    triage: TriageServiceDep,
) -> ReportSubmissionResponse:
    # 1. File the report. This commits before returning: the report, its case
    #    code and its first audit entry are durable from here on.
    result = service.submit(
        category=payload.category,
        description=payload.description,
        # HttpUrl is stored as its normalised string form.
        evidence_url=str(payload.evidence_url) if payload.evidence_url else None,
    )

    # 2. Then, and only then, ask the model for a suggestion. Deliberately
    #    after the commit and outside its transaction: a model that is missing,
    #    slow or broken must not be able to affect whether a report exists.
    #    run_for_report never raises — every failure it meets becomes a FAILED
    #    triage row — so nothing here can cost the reporter their case code.
    triage.run_for_report(result.report.id, payload.description)

    # 3. The response is unchanged from Phase 2. No triage data is returned to
    #    a reporter, ever; the moderator detail view is where it surfaces.
    return ReportSubmissionResponse(
        case_code=result.case_code,
        status=result.report.status,
        submitted_at=result.report.created_at,
    )

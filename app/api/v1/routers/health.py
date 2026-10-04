"""Health check endpoint.

Deliberately exposes only non-sensitive service metadata. It must stay cheap
and dependency-free so that a load balancer or uptime monitor can poll it
without touching the database or the ML model.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app import __version__
from app.core.config import Settings, get_settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Service liveness information."""

    status: Literal["ok"] = Field(description="Always 'ok' when the API is serving.")
    service: str = Field(description="Human-readable service name.")
    version: str = Field(description="Application version.")
    environment: str = Field(description="Deployment environment.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "service": "WhistleDrop AI",
                    "version": "0.1.0",
                    "environment": "development",
                }
            ]
        }
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns service liveness. Requires no authentication.",
)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        environment=settings.app_env,
    )

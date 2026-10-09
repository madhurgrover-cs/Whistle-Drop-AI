"""Application entry point.

Uses the application-factory pattern: ``create_app()`` builds a fully wired
FastAPI instance from a ``Settings`` object. Tests can therefore construct an
app with test configuration instead of mutating global state, and the module
stays free of import-time side effects beyond the default instance below.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import register_middleware

API_DESCRIPTION = """
**WhistleDrop AI** is a confidential reporting backend. Reports are submitted
without an account and without any identifying information. Each submission
returns a single-use **case code** that is the only way to track the report.

### Anonymity — what this system does and does not guarantee

* No reporter name, email, phone number or account is ever collected or stored.
* The case code is stored only as a keyed hash, never in plaintext.
* Network-level anonymity is **out of scope**. Use Tor or a VPN if your threat
  model requires it.
* A report can still de-anonymise its author through its own contents.

### Tracking a report

A successful submission returns a case code, once. It is stored only as a keyed
HMAC-SHA256 digest, so a lost code cannot be recovered or reissued by anyone.
Tracking uses `POST /cases/lookup` rather than a URL path, so the code stays out
of browser history, proxy logs and access logs.

### Moderator access

Reviewing reports requires a moderator account, obtained from an administrator —
there is no sign-up. `POST /api/v1/auth/login` exchanges credentials for a
short-lived bearer token. This changes nothing for reporters: submitting and
tracking a report remain anonymous and unauthenticated.

### AI-assisted triage

AI triage is an **extension** to the official task specification, not part of
it. Its output is advisory only: it can never change a report's status or its
official category. A moderator always decides.
"""


logger = logging.getLogger(__name__)


def _build_lifespan(settings: Settings):
    """Warm the triage model at startup, tolerantly.

    Loading the artifact costs a few seconds, nearly all of it importing
    scikit-learn beneath joblib. Doing it here rather than on the first report
    means that cost is paid while nobody is waiting, instead of by whichever
    reporter submits first after a restart.

    Every failure is caught. A missing, corrupt or incompatible artifact must
    never stop the application starting — filing a report is the one thing that
    has to work, and it does not need the model.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.ml_triage_enabled and settings.ml_warm_start:
            try:
                from app.api.deps import _triage_model

                model = _triage_model(settings.ml_artifact_root, settings.ml_model_version)
                if model.is_available():
                    logger.info("Triage model warmed at startup.")
                else:
                    logger.warning(
                        "Triage model could not be warmed; reports will still be filed "
                        "and triage will be recorded as FAILED."
                    )
            except Exception:
                # Never fatal. The reason is logged for the operator.
                logger.exception("Triage warm-up failed; continuing without it.")

        yield

    return lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure a FastAPI application instance."""
    settings = settings or get_settings()

    configure_logging(settings)

    app = FastAPI(
        # Never True. FastAPI's debug mode replaces unhandled errors with an
        # interactive traceback page that exposes source, locals and the
        # request — the settings' own `debug` flag deliberately does not reach
        # it, so a misconfigured deployment cannot turn tracebacks on.
        debug=False,
        title=settings.app_name,
        description=API_DESCRIPTION,
        version=__version__,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        contact={"name": "WhistleDrop AI", "url": "https://github.com/madhurgrover-cs"},
        license_info={"name": "MIT"},
        lifespan=_build_lifespan(settings),
    )

    # Registered before the routers so that every failure below — domain
    # error, validation failure or unhandled crash — leaves through the same
    # envelope and never carries internal detail with it.
    register_exception_handlers(app)

    # Security headers, CORS and the request-body ceiling. See
    # app/core/middleware.py for the ordering and why it is that way.
    register_middleware(app, settings)

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()

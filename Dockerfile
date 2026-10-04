# Production image for the WhistleDrop API.
#
# Three stages, so that the things needed to *build* the model never reach the
# image that serves traffic:
#
#   builder  installs runtime dependencies into a virtualenv
#   trainer  installs the training dependencies, runs the pipeline, produces
#            the model artifact, and is then discarded
#   runtime  copies the virtualenv and the artifact, and nothing else
#
# pandas, the training code and the dataset exist only in `trainer`. The final
# image carries the API, its dependencies and one 33 KB model file.
#
# The artifact is built rather than copied from the host because
# `ml/artifacts/` is gitignored — it is a build output, reproducible in about
# two seconds from the committed synthetic dataset and a fixed seed. Building
# it here means a clone of the repository produces a working image with no
# extra step, and the artifact in the image provably matches the code beside it.

# ---------------------------------------------------------------------------
# Stage 1: runtime dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: train the model artifact
# ---------------------------------------------------------------------------
FROM builder AS trainer

COPY requirements-ml.txt ./
RUN pip install -r requirements-ml.txt

# Only what training reads. Not the application, not the tests.
COPY ml/ ./ml/

RUN python -m ml.src.train_category \
    && test -f ml/artifacts/whistledrop-category-v1/category/model.joblib

# ---------------------------------------------------------------------------
# Stage 3: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# PYTHONDONTWRITEBYTECODE: the filesystem is read-only to the app user anyway.
# PYTHONUNBUFFERED: logs reach the collector immediately rather than on flush.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# A non-root user with no shell and no home directory to write into. If the
# process is ever compromised, it is compromised as a user that can do very
# little — it does not own the application files it runs.
RUN groupadd --system --gid 1001 whistledrop \
    && useradd --system --uid 1001 --gid whistledrop --no-create-home \
       --shell /usr/sbin/nologin whistledrop

WORKDIR /srv/whistledrop

COPY --from=builder /opt/venv /opt/venv

# Application source. Ownership stays with root: the runtime user needs to read
# these, never to modify them.
COPY --chown=root:root app/ ./app/
COPY --chown=root:root alembic/ ./alembic/
COPY --chown=root:root alembic.ini ./

# The inference-side helpers the API imports (text normalisation, keyword
# extraction, the priority heuristic). Not the trainer, not the dataset.
COPY --chown=root:root ml/__init__.py* ./ml/
COPY --chown=root:root ml/src/__init__.py ml/src/config.py ml/src/text.py \
     ml/src/keywords.py ml/src/priority.py ./ml/src/

# The trained artifact, from the stage that is about to be thrown away.
COPY --from=trainer --chown=root:root \
     /build/ml/artifacts/ ./ml/artifacts/

# Migrations are run as a separate step, not on startup: two replicas starting
# at once would otherwise race, and a deployment that migrates implicitly is a
# deployment that can migrate unintentionally. See docs/DEPLOYMENT.md.

USER whistledrop

EXPOSE 8000

# Hits the one endpoint that touches neither the database nor the model, so it
# answers "is this process serving?" and nothing else.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status==200 else 1)"

# --no-server-header: uvicorn adds `server: uvicorn` below the ASGI layer,
#   where no middleware can remove it. This is the only way to suppress it.
# --no-access-log: access logs record the client address. The application's own
#   logs carry the events without it. Remove this flag if you need them and
#   have somewhere appropriate to send them.
# --proxy-headers is deliberately NOT set: trusting X-Forwarded-For requires a
#   proxy that overwrites it. Enable it together with TRUST_PROXY_HEADERS only
#   when that is true.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--no-server-header", \
     "--no-access-log", \
     "--workers", "1"]

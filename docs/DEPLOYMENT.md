# Deployment

> **TLS is not optional.** Case codes and bearer tokens travel in request
> bodies and headers. Over plain HTTP both are readable in transit, and nothing
> in this application can compensate. Terminate TLS in front of it.

## What is in the image

`Dockerfile` builds in three stages so that the things needed to *build* the
model never reach the image that serves traffic:

| Stage | Contains | Kept? |
|---|---|---|
| `builder` | runtime dependencies in a virtualenv | copied forward |
| `trainer` | training deps, `ml/`, the dataset; runs the pipeline | **discarded** |
| `runtime` | app, alembic, the 5 inference modules, one 33 KB artifact | shipped |

Verified on the built image:

- runs as `uid=1001(whistledrop)`, not root
- **no `.env` anywhere on the filesystem**
- no `tests/`, no `ml/data/`, no `train_category.py`, no `generate_synthetic.py`
- `pandas` is not installed (training-only)

The model artifact is **built in the image** rather than copied from the host,
because `ml/artifacts/` is a gitignored build output. It is reproducible in
about two seconds from the committed synthetic dataset and a fixed seed, so a
clone produces a working image with no extra step — and the artifact provably
matches the code beside it.

## Running it

```bash
# Secrets come from the environment. Compose refuses to start without them
# rather than inventing a fallback.
export POSTGRES_USER=whistledrop
export POSTGRES_PASSWORD="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
export POSTGRES_DB=whistledrop
export CASE_CODE_PEPPER="$(python -c 'import secrets;print(secrets.token_hex(32))')"
export JWT_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')"

docker compose -f docker-compose.prod.yml up -d --build
```

This starts PostgreSQL, runs `alembic upgrade head` as a **separate one-shot
service**, and only then starts the API. Migrating in a separate step rather
than on application startup matters: two replicas starting together would
otherwise race to migrate the same database, and an application that migrates
implicitly is one that can migrate when nobody meant it to.

Then create the first moderator — there is no sign-up endpoint:

```bash
docker compose -f docker-compose.prod.yml exec api python -m scripts.create_moderator --username alex
```

## Hardening applied in `docker-compose.prod.yml`

| Control | Setting |
|---|---|
| Database not reachable from the host | no `ports:` on the `db` service |
| API bound to loopback only | `127.0.0.1:8000:8000` — put a proxy in front |
| Read-only root filesystem | `read_only: true` + a 16 MB `/tmp` tmpfs |
| No privilege escalation | `no-new-privileges:true` |
| All Linux capabilities dropped | `cap_drop: [ALL]` |
| No fallback secrets | `${VAR:?...}` — compose refuses to start |
| Debug off | `DEBUG=false`, and the app hard-wires `debug=False` regardless |

## Settings that need a decision

| Variable | Default | Set it when |
|---|---|---|
| `HSTS_MAX_AGE_SECONDS` | `0` (off) | TLS is actually terminated. Browsers ignore HSTS over plain HTTP, so sending it earlier implies a protection that does not exist. |
| `TRUST_PROXY_HEADERS` | `false` | A proxy in front **overwrites** `X-Forwarded-For`. Believing a spoofable header lets one client mint unlimited rate-limit identities. |
| `CORS_ALLOWED_ORIGINS` | empty | A browser front end exists. There is none today, so no CORS headers are emitted at all. |
| `ML_TRIAGE_ENABLED` | `true` | — Turning it off files reports exactly as before and writes no triage row. |

## Uvicorn flags, and why

The image's `CMD` sets:

- `--no-server-header` — uvicorn adds `server: uvicorn` *below* the ASGI layer,
  where no middleware can remove it. This is the only way to suppress it;
  verified on the wire.
- `--no-access-log` — the access log is the one place a client IP appears. The
  application's own logs carry the events without it. Remove this if you need
  access logs and have somewhere appropriate to send them.
- `--proxy-headers` is **not** set. Enable it together with
  `TRUST_PROXY_HEADERS=true`, and only behind a proxy that overwrites the header.

## Scaling beyond one process

`--workers 1` today. Two things need attention before raising it:

1. **Rate-limit counters are per process.** Four workers means four times the
   configured limit. `limits` already speaks Redis; this is configuration, not
   a code change.
2. **Each worker loads its own copy of the model.** 33 KB, so immaterial.

## Secret rotation

- Rotating `CASE_CODE_PEPPER` **permanently invalidates every case code already
  issued** — the digests cannot be re-derived. Treat it as long-lived.
- Rotating `JWT_SECRET_KEY` invalidates every access token in circulation
  immediately. That is the intended response to a suspected leak; moderators
  simply log in again.

## Operational checks

```bash
docker compose -f docker-compose.prod.yml ps          # both healthy?
docker compose -f docker-compose.prod.yml logs api    # startup + triage warm-up
curl -fsS http://127.0.0.1:8000/api/v1/health         # liveness
```

The health endpoint deliberately reports **only** that the process is serving.
It does not report database or model status: it is a public endpoint, and
publishing internal dependency state to anonymous callers tells an attacker
which part of the system is currently weak. Model availability is logged at
startup, and a moderator sees `triage.status = FAILED` on any report where
inference did not succeed.

## Not implemented

These are deployment responsibilities this repository does not attempt:

- TLS termination and certificate management
- Shared rate-limit storage across workers
- Database encryption at rest, backups, backup access control
- Log shipping, retention and access control
- Monitoring and alerting on 429/5xx rates
- Dependency vulnerability scanning in CI

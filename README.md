# WhistleDrop AI

![CI](https://github.com/YushBytes/WhistleDrop-AI/actions/workflows/ci.yml/badge.svg)

**Anonymous Reporting & Intelligent Case Triage System**

A confidential reporting backend. Anyone can submit a report **without an
account and without revealing their identity**, and receives a single secret
**case code** that is the only way to track it. Moderators triage the queue
without ever learning who reported what.

Built for **GDG on Campus SRM — Technical Recruitment 2026-27**
(task: *WhistleDrop — Speak Without Being Seen*).

> **Status: complete.** 751 tests passing, Ruff clean, migrations clean,
> end-to-end verified against a production Docker image.

---

## The problem

People who witness wrongdoing at work usually have good reasons not to report
it. The risk is not abstract — it is being identified, and then retaliated
against. Most reporting systems ask for an account, an email address, or "just
your name for follow-up", and every one of those is a link back to a person.

But a report nobody can follow up on is not much use either. The reporter needs
to know whether anything happened, and a moderator needs enough structure to
act on a queue.

## The solution

Collect **nothing** that identifies the reporter, and issue a single secret
code that is the only handle on the report.

- There is no `reporters` table. There is no column an identity could be
  written into. A submission carrying `email` or `name` is **rejected**, not
  quietly ignored.
- The case code is 100 bits of entropy, returned **once**, and stored only as
  an HMAC-SHA256 digest. Nobody can recover or reissue it — not a moderator,
  not an administrator, not someone holding the entire database.
- Moderators work the queue, change status, and write notes that are
  **internal by default**; publishing one to the reporter is always an explicit
  act.
- An AI model suggests a category and a priority to help moderators triage. It
  decides nothing, and it is structurally incapable of changing a report.

---

## Key features

| | |
|---|---|
| **Anonymous submission** | No account, no identifying field, closed request schema |
| **Secure case codes** | 100-bit entropy, Crockford Base32, HMAC-SHA256 at rest |
| **Case tracking** | POST-based lookup so the code never enters a URL or an access log |
| **Moderator auth** | bcrypt + HS256 JWT, no public sign-up, CLI-provisioned accounts |
| **Moderation queue** | Filter by status and category, paginated, deterministic ordering |
| **Status workflow** | Enforced transition map, terminal states, row-locked against races |
| **Audit trail** | Append-only `case_updates`, atomic with every status change |
| **Visibility control** | Internal vs published notes, filtered in SQL |
| **AI triage** | TF-IDF + logistic regression, advisory only, fully explainable |
| **Hardening** | Rate limiting, body-size ceiling, security headers, strict CORS |
| **Tests** | 751, against real PostgreSQL — never SQLite |

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
   REPORTER              │            FastAPI (app/)                │
   (anonymous)           │                                          │
       │                 │   ┌────────────────────────────────┐     │
       │  POST /reports  │   │ middleware                     │     │
       ├────────────────►│   │  security headers → CORS →     │     │
       │                 │   │  body-size limit               │     │
       │                 │   └───────────────┬────────────────┘     │
       │                 │                   ▼                      │
       │                 │   ┌────────────────────────────────┐     │
       │                 │   │ router (thin)                  │     │
       │                 │   │  + rate-limit dependency       │     │
       │                 │   └───────────────┬────────────────┘     │
       │                 │                   ▼                      │
       │                 │   ┌────────────────────────────────┐     │
       │                 │   │ Pydantic schema (closed)       │     │
       │                 │   └───────────────┬────────────────┘     │
       │                 │                   ▼                      │
       │                 │   ┌────────────────────────────────┐     │
       │                 │   │ service  (rules, transactions) │     │
       │                 │   └───────────────┬────────────────┘     │
       │                 │                   ▼                      │
       │                 │   ┌────────────────────────────────┐     │
       │                 │   │ repository  (SQL only)         │     │
       │                 │   └───────────────┬────────────────┘     │
       │                 └───────────────────┼──────────────────────┘
       │                                     ▼
       │                    ┌────────────────────────────────────┐
       │                    │          PostgreSQL 16             │
       │   case code        │                                    │
       │◄───────────────────┤  ┌──────────────────────────────┐  │
          (once, never      │  │ OFFICIAL RECORD              │  │
           recoverable)     │  │   reports                    │  │
                            │  │   case_updates  (audit)      │  │
                            │  │   moderators                 │  │
                            │  └──────────────────────────────┘  │
                            │                                    │
                            │  ┌──────────────────────────────┐  │
                            │  │ AI ADVISORY  (separate)      │  │
                            │  │   report_triage              │  │
                            │  └──────────────▲───────────────┘  │
                            └─────────────────┼──────────────────┘
                                              │ after COMMIT,
                                              │ never in the same txn
                            ┌─────────────────┴──────────────────┐
                            │   app/ml  —  the ML boundary       │
                            │   TF-IDF → LogisticRegression      │
                            │   + deterministic priority rule    │
                            └────────────────────────────────────┘
                                              │
                                              ▼
                                        MODERATOR
                                    (authenticated)
                                              │
                                              ▼
                                     HUMAN DECISION
                                   (the only authority)
```

**The separation that matters.** The official record and the AI's opinion live
in *different tables*. `reports.category` and `reports.status` are set by a
human and read by the moderation workflow; `report_triage` is written by the
model and read by nothing but the moderator detail view. The AI cannot change
a report because there is no code path from one to the other.

### Layering

```
router → service → repository → PostgreSQL
```

Strictly one-directional. Routers never touch the ORM; services never touch
HTTP; repositories never open transactions. `app/ml/` is the only place
scikit-learn, joblib or numpy appear — asserted by a test that walks the AST of
every file under `app/`.

---

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI (Python 3.12) | Typed, async-capable, generates OpenAPI |
| Validation | Pydantic v2 | Closed schemas — unknown fields rejected |
| Database | PostgreSQL 16 | Native enums, JSONB, `SELECT … FOR UPDATE` |
| ORM | SQLAlchemy 2.0 (typed) | Expression language, no string SQL |
| Migrations | Alembic | Schema reproducible from zero |
| Auth | bcrypt + PyJWT (HS256) | Direct, no unmaintained wrapper |
| Rate limiting | `limits` | The engine slowapi wraps, without its decorator |
| ML | scikit-learn (TF-IDF → LogReg) | Explainable, fast, auditable |
| Tests | pytest against real PostgreSQL | Never SQLite |
| Lint/format | Ruff | Includes bandit security rules |

Deliberately **not** used: any LLM or AI API, embeddings, a vector database,
TensorFlow, PyTorch, Redis, Celery, Kafka. Nothing on this roadmap needs them.

---

## API endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /api/v1/reports` | — | File a report anonymously. Returns the case code **once**. |
| `POST /api/v1/cases/lookup` | — | Track a report using that code. |
| `POST /api/v1/auth/login` | — | Moderator login. Returns a bearer token. |
| `GET /api/v1/moderation/reports` | **Bearer** | Review queue: filter, paginate. |
| `GET /api/v1/moderation/reports/{id}` | **Bearer** | One report in full, with its history and AI triage. |
| `PATCH /api/v1/moderation/reports/{id}/status` | **Bearer** | Move a report through the workflow. |
| `GET /api/v1/health` | — | Liveness. No database, no model. |

Interactive documentation at **`/docs`** (Swagger) and **`/redoc`**.

### Authentication

Moderators only. `POST /api/v1/auth/login` exchanges a username and password
for an **HS256 JWT** valid for 30 minutes, sent as `Authorization: Bearer
<token>`.

- **No public sign-up, by design.** Accounts are created by an administrator
  running `python -m scripts.create_moderator`. A whistleblowing system whose
  moderator set can be joined over HTTP is not one anybody should file into.
- Passwords are hashed with **bcrypt** (cost 12). Passwords over bcrypt's
  72-byte limit are **rejected, not silently truncated** — the pinned version
  truncates, which would let two different passwords authenticate.
- Unknown username, wrong password and deactivated account return one
  **identical 401**, in about the same time (an unknown username still pays for
  a bcrypt comparison, so timing cannot enumerate accounts).
- A deactivated moderator loses access on their **next request**, not at token
  expiry — the account is re-read every time.

### The case-code mechanism

```
WD-4K7PQ-92MRT-XJ3HN-B8VZ6
   └─ 20 chars of Crockford Base32 = 100 bits of entropy, from `secrets`
```

Crockford Base32 excludes `I`, `L`, `O` and `U`, so there is no character pair
to misread. Lookup accepts any case, spacing or hyphenation.

Only `HMAC-SHA256(code, CASE_CODE_PEPPER)` is stored. **A lost case code cannot
be recovered**: any recovery route would require knowing who filed the report,
which is exactly what the system refuses to record.

HMAC rather than bcrypt is deliberate — a case code is 100 uniformly random
bits, so there is nothing to brute-force. What the system needs is a
*deterministic* digest it can look up on a unique index; a salted password hash
would force a full table scan and a slow verify per row.

**Lookup is a POST.** A case code is a bearer credential, and anything in a URL
lands in browser history, `Referer` headers, proxy and access logs, and CDN
telemetry. A request body appears in none of them. One broken REST convention
is the right trade.

---

## Database design

```
reports ──1:1── report_triage        AI suggestions; deletes WITH the report
   │
   └──1:N── case_updates             append-only audit trail; BLOCKS deletion
                  │
moderators ──1:N──┘                  actor nulled if the account is deleted
```

**There is no `reporters` table**, and a test asserts it — along with the
absence of `reporter_id`, `email`, `name`, `phone` and `ip_address` on every
table.

| Table | Holds |
|---|---|
| `reports` | Category, description, optional evidence URL, status, `case_code_hash` |
| `case_updates` | One row per status change. Written, never edited |
| `report_triage` | The model's suggestion. Never authoritative |
| `moderators` | Staff accounts — the only identified parties in the system |

UUID primary keys throughout: a sequential id would leak how many reports exist
and in what order they arrived.

Delete behaviour is chosen per relationship: triage is `CASCADE` (derived data,
owned by its report), audit entries are **`RESTRICT`** (history is not
collateral damage of a `DELETE`), and a moderator is `SET NULL` (history
outlives the account).

---

## AI-assisted triage

> ⚠ **`whistledrop-category-v1` was trained on SYNTHETIC data.** No labelled
> corpus of real whistleblowing reports exists, so the training set is
> generated from templates. Its evaluation scores measure how well a linear
> model recovers a template generator — an easy problem — and **do not
> establish real-world classification accuracy**. Every suggestion is advisory.
> See [`ml/reports/model-card.md`](ml/reports/model-card.md).

```
report text → TF-IDF (word 1–2 grams) → logistic regression → suggested category + probability
```

### Official category vs AI suggestion

| | Field | Set by |
|---|---|---|
| **Official** | `reports.category` | The reporter, then a moderator |
| **Suggestion** | `report_triage.suggested_category` | The model — advisory only |

### Why this model

TF-IDF and logistic regression, not embeddings and not an LLM. Every prediction
decomposes into **exact term contributions** that can be shown to a moderator.
In a system where a suggestion sits beside a confidential report, being able to
say precisely which words moved a decision is worth more than a few points of
accuracy from something opaque. **No report text is ever sent anywhere.**

### Priority is a heuristic, not a model

There are no priority labels to train on. Inventing some would mean the model
learned whatever rule invented them; using time-to-resolution as a proxy would
encode past staffing rather than severity. So `LOW`/`MEDIUM`/`HIGH`/`CRITICAL`
comes from a documented deterministic rule that reports which signals fired.
It is always described as a *heuristic priority suggestion*.

### Human-in-the-loop

```
AI suggestion → human review → human decision → curated dataset → future model version
```

A moderator who disagrees simply sets the category they judge correct. Nothing
pushes back and **nothing retrains automatically**. That disagreement is the
most valuable signal the system can produce, and curating it — deliberately, by
a person — is what will eventually replace the synthetic data.

### ML failure isolation

Inference runs **after** the submission transaction commits, outside it. A
missing artifact, a corrupt one, a crash, or output that fails validation all
produce a `FAILED` triage row — and the reporter still receives their case
code, with the report, audit entry and status exactly as they would have been.

Cost: ~1 ms per report once warm. The artifact loads at startup so no reporter
pays the ~3 s scikit-learn import.

---

## Security measures

Full control-by-control detail, including what is *not* implemented, in
[`docs/security-checklist.md`](docs/security-checklist.md).

| Area | Measure |
|---|---|
| Rate limiting | Reports 5/min, lookup 10/min, login 5/min — **salted** IP digests, never stored |
| Request size | 64 KiB ceiling, enforced before parsing; chunked bodies counted as they stream |
| Headers | `nosniff`, `no-referrer`, `DENY`, CSP, Permissions-Policy, COOP/CORP |
| CORS | **No origins by default** — there is no browser front end |
| Errors | One envelope; no stack trace, SQL, DSN or library message ever |
| Secrets | Required, no defaults, `SecretStr`, placeholder-rejecting |
| Logging | Events, not values — verified by parsing every `logger.*` call site |
| SQL | Expression language only; no dynamic SQL anywhere (AST-verified) |
| Concurrency | `SELECT … FOR UPDATE`; verified to fail *without* the lock |

### What "anonymous" means here, precisely

This service collects, stores and derives **no reporter identity**.

That is an *application-level* guarantee, and not the same as anonymity. Out of
scope, with no claim made:

- **network layer** — the reporter's IP reaches whatever terminates TLS, and
  infrastructure in front commonly logs it. Tor or a VPN is the reporter's
  answer, not ours;
- **traffic analysis** — submission timing and request size are observable;
- **the report's own contents** — stored exactly as written, because it is
  evidence, and it can identify its author;
- **writing style** — stylometry works;
- **operational access** — an administrator, a backup or a legal order reaches
  the stored rows.

The project does not claim to be "100% anonymous" or "fully secure", and a test
asserts the API documentation makes no such claim.

---

## Setup

### Prerequisites

- **Python 3.12** (3.13+ unsupported — see `pyproject.toml`)
- **Docker Desktop** — for PostgreSQL

### 1. Install

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell
# source .venv/bin/activate           # macOS / Linux

pip install -r requirements.txt -r requirements-dev.txt -r requirements-ml.txt
```

### 2. Configure

```bash
copy .env.example .env                # PowerShell
# cp .env.example .env                # macOS / Linux
```

Then edit `.env` and set three values:

```bash
# A strong database password, used consistently in DATABASE_URL and TEST_DATABASE_URL
POSTGRES_PASSWORD=...

# Two DIFFERENT random keys:
python -c "import secrets; print(secrets.token_hex(32))"   # → CASE_CODE_PEPPER
python -c "import secrets; print(secrets.token_hex(32))"   # → JWT_SECRET_KEY
```

The application **refuses to start** without these, refuses the placeholders
shipped in `.env.example`, and refuses anything under 32 characters.

`.env` is gitignored and must never be committed.

### 3. Databases

```bash
docker compose up -d
docker compose ps        # both should report "healthy"
```

Two containers: development on **5432**, and a disposable in-memory test
database on **5433**, so the test suite can never touch development data.

### 4. Migrate

```bash
alembic upgrade head
```

### 5. Train the model

```bash
python -m ml.src.train_category
```

Takes about two seconds. `ml/artifacts/` is gitignored — a trained model is a
build output, reproducible from the committed synthetic dataset and a fixed
seed. **Skipping this is safe**: reports are still filed and triage simply
records `FAILED`.

### 6. Create a moderator

```bash
python -m scripts.create_moderator --username demo.moderator
```

The password is prompted for without echo and never accepted as an argument —
an argument lands in shell history and the process list.

### 7. Run

```bash
uvicorn app.main:app --reload
```

| URL | |
|---|---|
| <http://127.0.0.1:8000/docs> | Swagger UI |
| <http://127.0.0.1:8000/redoc> | ReDoc |
| <http://127.0.0.1:8000/api/v1/health> | Health check |

Outside local development, suppress uvicorn's own `Server` header — it is added
below the ASGI layer, where no middleware can remove it:

```bash
uvicorn app.main:app --no-server-header --no-access-log
```

---

## Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `DATABASE_URL` | **yes** | — | No fallback: deploying without a database must fail loudly |
| `CASE_CODE_PEPPER` | **yes** | — | HMAC key for case codes. Rotating invalidates every issued code |
| `JWT_SECRET_KEY` | **yes** | — | Token signing key. Must differ from the pepper |
| `TEST_DATABASE_URL` | for tests | — | Must end in `_test` or the suite refuses to run |
| `APP_ENV` | no | `development` | `production` enforces a bcrypt cost ≥ 12 |
| `LOG_LEVEL` | no | `INFO` | |
| `JWT_ALGORITHM` | no | `HS256` | HMAC family only |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | no | `30` | No refresh token: expiry means logging in again |
| `RATE_LIMIT_ENABLED` | no | `true` | |
| `RATE_LIMIT_REPORTS` / `_CASE_LOOKUP` / `_LOGIN` | no | `5/minute;30/hour`, `10/minute;60/hour`, `5/minute;20/hour` | |
| `TRUST_PROXY_HEADERS` | no | `false` | Only behind a proxy that **overwrites** `X-Forwarded-For` |
| `MAX_REQUEST_BODY_BYTES` | no | `65536` | |
| `CORS_ALLOWED_ORIGINS` | no | empty | Comma-separated. Empty = no CORS headers at all |
| `HSTS_MAX_AGE_SECONDS` | no | `0` | Set only where TLS is terminated |
| `ML_TRIAGE_ENABLED` | no | `true` | |
| `ML_MODEL_VERSION` | no | `whistledrop-category-v1` | Never `latest`, never request-selectable |
| `ML_WARM_START` | no | `true` | Load the model at startup, not on the first report |

---

## Running tests

```bash
pytest                          # 751 tests against real PostgreSQL
pytest --cov --cov-report=term-missing
ruff check . && ruff format --check .
alembic check                   # models and schema agree
```

The integration tests rebuild the test database from migrations on every run,
and **refuse to start** unless the target database's name ends in `_test`.

See [`docs/COVERAGE.md`](docs/COVERAGE.md) — 94%, with the gaps explained.

---

## Example requests

**File a report** (no authentication):

```bash
curl -sX POST http://127.0.0.1:8000/api/v1/reports \
  -H 'Content-Type: application/json' \
  -d '{"category":"SECURITY",
       "description":"Production database credentials are in a public chat channel.",
       "evidence_url":"https://example.com/screenshot"}'
```

```json
{
  "case_code": "WD-4K7PQ-92MRT-XJ3HN-B8VZ6",
  "status": "SUBMITTED",
  "submitted_at": "2026-09-25T10:33:34.894504Z",
  "message": "Save this case code somewhere safe. It is the only way to check your report, and it cannot be recovered or reissued if you lose it."
}
```

**Track it** (no authentication):

```bash
curl -sX POST http://127.0.0.1:8000/api/v1/cases/lookup \
  -H 'Content-Type: application/json' \
  -d '{"case_code":"WD-4K7PQ-92MRT-XJ3HN-B8VZ6"}'
```

```json
{
  "status": "UNDER_REVIEW",
  "category": "SECURITY",
  "submitted_at": "2026-09-25T10:33:34.894504Z",
  "last_updated_at": "2026-09-25T11:02:14.907000Z",
  "timeline": [
    {"status": "SUBMITTED", "note": "Report received. It is queued for review by a moderator.", "occurred_at": "..."}
  ]
}
```

**Log in and move a report:**

```bash
TOKEN=$(curl -sX POST http://127.0.0.1:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo.moderator","password":"..."}' | jq -r .access_token)

curl -s "http://127.0.0.1:8000/api/v1/moderation/reports?status=SUBMITTED" \
  -H "Authorization: Bearer $TOKEN"

curl -sX PATCH "http://127.0.0.1:8000/api/v1/moderation/reports/<id>/status" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"status":"UNDER_REVIEW","note":"Confirmed.","visible_to_reporter":false}'
```

## Example workflow

One command walks the whole thing and asserts every privacy guarantee:

```bash
python -m scripts.demo
```

It files a report, tracks it, logs a moderator in, shows the AI suggestion
beside the official category, moves `SUBMITTED → UNDER_REVIEW` with an internal
note, proves the reporter cannot see it, moves `→ RESOLVED` with a published
note, proves the reporter *can* see that one, and confirms an illegal
transition is refused. It exits non-zero if any check fails.

---

## Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

Three-stage build: the trainer stage produces the model and is discarded, so
the runtime image has no pandas, no dataset and no training code. Runs as a
non-root user on a read-only filesystem with all capabilities dropped. **`.env`
is never copied into the image** — secrets come from the environment.

> **TLS is not optional.** Case codes and bearer tokens travel in bodies and
> headers; over plain HTTP both are readable in transit.

---

## Known limitations

Stated plainly rather than buried.

1. **The AI model is trained on synthetic data.** Its metrics describe a
   template generator. This subsumes every other ML limitation and is the most
   important thing to know about the system.
2. **Model confidence is uncalibrated.** A 0.9 does not mean "right nine times
   in ten".
3. **No fairness evaluation is possible.** The system collects no demographic
   data — deliberately — which also forecloses measuring differential
   performance. Bias is unmeasured, not absent.
4. **Rate-limit counters are per process.** Four workers means four times the
   limit. `limits` already speaks Redis; this is configuration.
5. **Everyone behind one NAT or VPN exit shares a rate-limit counter.**
   Reporters using Tor are the most likely to be throttled by other traffic.
6. **No retry for failed triage.** A moderator sees `FAILED` and works unaided.
   Judged not worth the infrastructure at this scale.
7. **Network-layer anonymity is out of scope**, as described above.
8. **English only.**
9. **One moderator role.** No hierarchy, no per-report assignment.
10. **`updated_at` is maintained by the ORM**, so a raw SQL `UPDATE` bypasses it.

## Future improvements

1. **Replace the synthetic dataset with real, moderator-labelled reports.**
   Nothing else on this list matters as much. The override loop already exists;
   curating it is the work.
2. Calibration analysis, so the confidence shown to moderators means what it
   appears to mean.
3. Shared rate-limit storage (Redis) for multi-worker deployments.
4. Moderator roles and per-report assignment.
5. A retry path for failed triage, moderator-triggered.
6. Report attachments, held separately from report text.
7. Reporter-facing message replies, if it can be done without weakening
   anonymity.

---

## Project structure

```
WhistleDrop/
├── app/
│   ├── main.py              application factory, lifespan
│   ├── core/                config, errors, security, middleware, logging
│   ├── api/v1/routers/      HTTP route handlers (thin)
│   ├── schemas/             Pydantic request/response models
│   ├── services/            business logic, transactions
│   ├── repositories/        database access
│   ├── models/              SQLAlchemy ORM models
│   ├── db/                  base, engine, session
│   └── ml/                  the ML boundary: artifact loading + inference
├── ml/                      offline training pipeline (data, src, reports)
├── scripts/                 moderator seeding, end-to-end demo
├── alembic/                 migrations
├── docs/                    security checklist, deployment, coverage
├── tests/                   751 tests
├── Dockerfile               three-stage production image
├── docker-compose.yml       development databases
└── docker-compose.prod.yml  production stack
```

---

## Roadmap

- [x] **Phase 0** — Repository safety, scaffolding, config, health endpoint, Swagger
- [x] **Phase 1** — SQLAlchemy models, enums, Alembic migrations, PostgreSQL
- [x] **Phase 2** — Anonymous submission, case-code generation, case tracking
- [x] **Phase 3** — Moderator authentication (JWT, seeded accounts)
- [x] **Phase 4** — Moderation workflow, filtering, status transitions, audit trail
- [x] **Phase 5** — Rate limiting, error handling, security hardening
- [x] **Phase 6** — ML dataset and training pipeline (offline)
- [x] **Phase 7** — ML integration into the API (advisory triage on submission)
- [x] **Phase 8** — Production readiness, security audit, documentation

---

## License

MIT — see [`LICENSE`](LICENSE).

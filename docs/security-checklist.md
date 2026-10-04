# Security checklist

Status of every control, as of **Phase 8 of 8 (final)**.

A box is ticked only where the control exists in the code **and** a test
exercises it. Anything planned, partial, or left to deployment is listed
explicitly as such — an inaccurate checklist is worse than none.

| Symbol | Meaning |
|---|---|
| ☑ | Implemented and tested |
| ☐ | Not implemented |
| ⚙ | Deployment responsibility — the code supports it, the operator must configure it |

---

## Authentication

- [x] Passwords hashed with bcrypt (cost 12), never stored in plaintext
- [x] Verification uses the library's own constant-time comparison
- [x] Passwords over bcrypt's 72-byte limit **rejected**, not silently truncated
- [x] Passwords NFC-normalised so the same input verifies across keyboards
- [x] Minimum length 12; no composition rules (NIST SP 800-63B)
- [x] Access tokens are signed JWTs, HS256, 30-minute expiry
- [x] Algorithm restricted to the HMAC family in configuration and at verification
- [x] `alg: none` and algorithm-confusion tokens rejected
- [x] Token type claim checked, so one token kind cannot be replayed as another
- [x] Required claims enforced by the library (`sub`, `exp`, `iat`, `type`)
- [x] No clock-skew leeway (one service issues and verifies)
- [x] Deactivated moderators cannot log in
- [x] Deactivated moderators lose access on their **next request**, not at expiry
- [x] Unknown username, wrong password and deactivated account are indistinguishable
- [x] Timing equalised on unknown usernames (dummy bcrypt verification)
- [x] `WWW-Authenticate` challenge on 401, with no `error_description`
- [x] No public registration endpoint; accounts exist only via the seeding CLI
- [x] Seeding CLI refuses a `--password` argument (shell history, `ps`)
- [ ] Refresh tokens — deliberately absent; expiry means logging in again
- [ ] Token revocation list — `jti` is issued to make this possible later
- [ ] Multi-factor authentication
- [ ] Moderator roles — one role today; not needed yet

## Authorization

- [x] Every moderation endpoint requires `CurrentModerator`
- [x] Public reporting endpoints require no credential and accept none
- [x] `moderator_id` cannot be supplied by a client — rejected, not ignored
- [x] The acting moderator is always taken from the verified token
- [x] Status transitions validated server-side against an explicit map
- [x] Terminal states (`RESOLVED`, `DISMISSED`) cannot be reopened
- [x] OpenAPI marks moderation endpoints as authenticated and public ones as not

## Secrets

- [x] `CASE_CODE_PEPPER` and `JWT_SECRET_KEY` required, with **no** defaults
- [x] Both refuse the placeholder shipped in `.env.example`
- [x] Both refuse values shorter than 32 characters
- [x] Both are `SecretStr` — masked in reprs, `str()`, dumps and JSON
- [x] The two keys must differ (a leak of one is not a leak of both)
- [x] `.env` is gitignored and verified untracked by a test
- [x] A test scans every tracked and untracked file for the real local secrets
- [x] Production refuses a test-grade bcrypt work factor
- [ ] Secret rotation procedure — rotating either key invalidates issued
      case codes or tokens respectively; documented, not automated
- [ ] A managed secret store (Vault, cloud KMS) — ⚙ deployment choice

## Logging

- [x] Call sites log **events, not values** — verified by parsing the source
- [x] No `print()` anywhere under `app/` — verified by parsing the source
- [x] Case codes never logged, in any path (verified by driving real traffic)
- [x] `case_code_hash` never logged
- [x] Report descriptions, notes and evidence URLs never logged
- [x] Passwords, password hashes and tokens never logged
- [x] `CASE_CODE_PEPPER` and `JWT_SECRET_KEY` never logged
- [x] SQL echo off unconditionally, in every environment
- [x] `hide_parameters=True` on the engine — without it a failed `INSERT`
      raises an error whose text embeds the report body, and the unhandled-error
      handler would write it to disk
- [x] Alembic's `fileConfig` no longer disables application loggers
      (`disable_existing_loggers=False`) — it silently did
- [x] The rate limiter logs the throttling event but never the caller
- [ ] Structured (JSON) logging — ⚙ deployment choice
- [ ] Log shipping, retention and access control — ⚙ deployment responsibility

### Logging limitations, stated plainly

- Uvicorn's **access log** records method, path, status and client address.
  It is written by the server, not this application, and is disabled with
  `--no-access-log`. Note what it *cannot* contain: case codes travel in
  request bodies, never in URLs — the reason `/cases/lookup` is a POST — so no
  access log line has ever held one.
- **Report ids** appear in moderation paths and therefore in access logs. They
  are internal identifiers and say nothing about who filed a report.
- Logs are not an anonymity mechanism and are not claimed to be one.

## Input validation

- [x] Request schemas are closed (`extra="forbid"`) — unknown fields rejected
- [x] Every identity-shaped field rejected on submission, not silently dropped
- [x] Description bounded 10–20,000 characters; whitespace-only rejected
- [x] NUL bytes rejected (PostgreSQL text cannot hold them)
- [x] Evidence URL validated as http(s), bounded at 2,048 characters
- [x] Evidence URLs are **never fetched, opened or inspected** by this service
- [x] Category and status constrained by enums at the schema and in the database
- [x] Report ids must parse as UUIDs before reaching the database
- [x] Request bodies capped at 64 KiB, enforced before parsing
- [x] Bodies without `Content-Length` counted as they stream, not buffered
- [x] Malformed content types rejected without a traceback

## Database safety

- [x] All access through SQLAlchemy's expression language
- [x] No dynamic SQL construction — verified by parsing the source for
      `text()`/`execute()` calls with non-literal arguments
- [x] Injection payloads stored as text, never executed (tested)
- [x] Filters bound as parameters; filter values constrained by enums
- [x] UUID primary keys — no enumerable sequential ids
- [x] Constraints enforced in the database, not only in the application
- [x] Status change and its audit entry are atomic
- [x] Row locking (`SELECT … FOR UPDATE`) prevents contradictory concurrent state
- [x] Audit trail is append-only; existing rows are never modified
- [x] `case_updates.report_id` is `ON DELETE RESTRICT` — history blocks deletion
- [ ] Encryption at rest — ⚙ deployment responsibility (disk or cluster level)
- [ ] Separate least-privilege database roles for app and migrations — ⚙

## Case-code security

- [x] 100 bits of entropy, drawn from `secrets` (never `random`)
- [x] Crockford Base32 — no ambiguous characters to mistranscribe
- [x] Stored **only** as `HMAC-SHA256(code, pepper)`; plaintext never persisted
- [x] Verified by scanning every text column of every table for the plaintext
- [x] Deterministic digest enables a single indexed lookup
- [x] Returned exactly once, at submission; unrecoverable if lost
- [x] Submitted in a request **body**, never a URL
- [x] Lookup responses are `Cache-Control: no-store`
- [x] Malformed and unknown codes produce byte-identical 404s
- [x] Malformed codes are rejected before any query runs
- [x] Internal notes filtered in SQL, never loaded into a reporter's response
- [x] Report description never returned by case lookup

## Rate limiting

- [x] `POST /reports` — 5/minute, 30/hour
- [x] `POST /cases/lookup` — 10/minute, 60/hour
- [x] `POST /auth/login` — 5/minute, 20/hour (failed attempts count)
- [x] Implemented as a route dependency; services are untouched
- [x] Limits are per endpoint — exhausting one does not lock out another
- [x] `429` uses the standard error envelope, with `Retry-After`
- [x] The response leaks no limit, window, remaining count or caller identity
- [x] Counter keys are salted digests — the address is never stored or logged
- [x] `X-Forwarded-For` ignored unless `TRUST_PROXY_HEADERS` is set
- [x] A malformed limit expression logs loudly and disables the limit rather
      than taking the reporting endpoint offline
- [x] Moderation endpoints deliberately not limited (authenticated, accountable)
- [ ] Shared storage across workers — ⚙ counters are per process; `limits`
      already speaks Redis, so this is configuration, not a code change

### Rate-limiting limitations

- **Per process.** Four workers means four times the configured limit, and a
  restart resets the windows.
- **Shared addresses.** Everyone behind one NAT or VPN exit shares a counter.
  For this service that matters: reporters using Tor or a VPN are the most
  likely to be throttled by someone else's traffic. The limits are set loosely
  enough that ordinary use is unaffected.
- **Not anonymity.** Rate limiting neither hides who is calling nor stops a
  determined caller spreading requests across addresses.
- **Not a defence against case-code guessing** in any meaningful sense — 100
  bits already makes guessing hopeless. It is cost control.

## CORS

- [x] No origins allowed by default — this API has no browser front end
- [x] No CORS headers emitted at all when unconfigured
- [x] Allowed origins are an explicit list from configuration
- [x] Credentials disabled by default (bearer header, not cookies)
- [x] Wildcard origin **with** credentials refused at startup
- [x] Methods and headers restricted to those actually used
- [x] Preflight verified for both allowed and disallowed origins

**CORS is not an authentication mechanism.** It is a policy a *browser*
enforces on *other web pages*. It does nothing about `curl`, a script, or any
non-browser client — those are governed by the bearer token.

## Security headers

- [x] `X-Content-Type-Options: nosniff`
- [x] `Referrer-Policy: no-referrer`
- [x] `X-Frame-Options: DENY` and CSP `frame-ancestors 'none'`
- [x] `Content-Security-Policy` — `default-src 'none'` for the API; a separate
      policy for `/docs`, which must load Swagger
- [x] `Permissions-Policy` disabling device access
- [x] `Cross-Origin-Opener-Policy` / `Cross-Origin-Resource-Policy`
- [x] `Server` header does not name the stack
- [x] Present on health, public, authenticated **and error** responses
- [x] Present on the 500 generated outside the middleware stack
- [x] `Cache-Control: no-store` on every response carrying report or token data
- [ ] `Strict-Transport-Security` — ⚙ off by default and **correctly so**:
      browsers ignore HSTS over plain HTTP, so emitting it in development
      would imply a protection that does not exist. Set `HSTS_MAX_AGE_SECONDS`
      only where TLS is actually terminated.

## Error handling

- [x] One error envelope for every failure
- [x] Unhandled exceptions become a flat 500 with a fixed message
- [x] No stack trace, SQL, DSN, filesystem path or library message in any response
- [x] FastAPI debug mode hard-wired off, unreachable from configuration
- [x] Validation errors report the field and reason, never the rejected value
- [x] Authentication failures remain uniform after hardening (tested)
- [x] Case-lookup failures remain uniform after hardening (tested)
- [x] Swept across every route with a dozen shapes of bad input

## Privacy limitations

These are properties of the system as designed, not defects:

- **Application-level non-collection is what is guaranteed.** No reporter name,
  email, phone, account or address is collected, stored or derived. There is no
  reporters table and no column one could be written into.
- **Network-level anonymity is out of scope.** The reporter's IP reaches
  whatever terminates TLS, and infrastructure in front of this service commonly
  logs it. Tor or a VPN is the reporter's answer, not ours.
- **Traffic analysis.** Submission timing and request size are observable to
  anyone positioned on the network.
- **Report contents.** A report can identify its author through what it says.
  The text is stored exactly as written, because it is evidence.
- **Writing style.** Stylometry works, and a long report is a large sample.
- **Operational access.** A database administrator, a backup, or a legal order
  reaches the stored rows. What those rows contain is the guarantee; that they
  are reachable is assumed.

The project does not claim to be "100% anonymous", "completely anonymous" or
"fully secure", and a test asserts the API documentation makes no such claim.

## Container and deployment

Implemented in `Dockerfile` and `docker-compose.prod.yml`, and **verified on the
built image** (see `docs/DEPLOYMENT.md`):

- [x] Runs as a non-root user — verified `uid=1001(whistledrop)` in the image
- [x] `.env` is never copied in — verified absent from the whole filesystem
- [x] No development credentials baked in; every secret from the environment
- [x] Compose refuses to start without secrets (`${VAR:?...}`), no fallbacks
- [x] Database publishes no ports; reachable only on the compose network
- [x] API bound to loopback, expecting a TLS-terminating proxy in front
- [x] Read-only root filesystem, `no-new-privileges`, all capabilities dropped
- [x] Healthcheck on the one endpoint that touches neither DB nor model
- [x] `--no-server-header` in the production command (uvicorn's header is added
      below ASGI, where no middleware can remove it)
- [x] `--no-access-log` — the one place a client IP would appear
- [x] Debug mode off, and unreachable from configuration
- [x] Migrations run as a separate one-shot service, not on startup — two
      replicas would otherwise race
- [x] Training code, dataset and pandas excluded from the runtime image
- [x] Model artifact built in a discarded stage, so it matches the code shipped
      beside it
- [x] Schema verified to build from **zero** on an empty database, with all
      tables, enums, constraints, indexes and FK delete rules correct

Still the operator's responsibility:

- [ ] **TLS termination is required.** Case codes and bearer tokens travel in
      request bodies and headers; over plain HTTP both are readable in transit.
      Nothing in this application can compensate for its absence. ⚙
- [ ] Set `HSTS_MAX_AGE_SECONDS` once TLS is in place ⚙
- [ ] Set `TRUST_PROXY_HEADERS=true` **only** behind a proxy that overwrites
      `X-Forwarded-For` ⚙
- [ ] Shared rate-limit storage if running more than one worker ⚙
- [ ] Database encryption at rest, backups, and backup access control ⚙
- [ ] Log shipping, retention and access control ⚙
- [ ] Monitoring and alerting on 429 and 5xx rates ⚙
- [ ] Dependency vulnerability scanning in CI ⚙

## AI triage (Phase 7)

- [x] Model version pinned (`whistledrop-category-v1`), never `latest`
- [x] Version and artifact path are server-controlled; a request naming either
      is rejected 422 by the closed schema
- [x] No user-supplied paths reach `joblib.load`; no uploads, no dynamic imports
- [x] Missing, corrupt or incompatible artifact does not prevent startup
- [x] An artifact declaring a different version than the pinned one is refused
- [x] Inference runs **after** the submission transaction commits, outside it
- [x] Inference failure never rolls back a report — 5 failure modes tested
- [x] Invalid model output (unknown category, confidence out of range, NaN,
      mis-shaped vector) is refused, not clamped, and recorded as `FAILED`
- [x] Keywords sanitised: bounded, deduplicated, strings only
- [x] `report_triage` is one-to-one; retries update the row, never duplicate it
- [x] Reporter-facing endpoints expose no triage data at all
- [x] Moderators see triage, clearly labelled as a suggestion
- [x] AI cannot modify `reports.category`, `reports.status`, or the timeline
- [x] Priority documented everywhere as a **heuristic**, not a trained model
- [x] Keywords documented as **model-associated**, never as causal explanations
- [x] Synthetic-data warning in the model card, evaluation report, README,
      OpenAPI moderator docs, and the artifact metadata
- [x] Inference is local and offline — no external AI SDK is installed or
      importable from `app/ml/` (AST-asserted); a test blocks every socket
- [x] Report text is never logged on the inference path
- [ ] Retry for failed triage — judged not worth the infrastructure; a
      moderator sees `FAILED` and works unaided. Documented as a limitation.

---

*Final as of Phase 8. Every ticked box has both an implementation and a test;
every unticked one is either a deliberate decision with a stated reason or an
operator responsibility marked ⚙.*

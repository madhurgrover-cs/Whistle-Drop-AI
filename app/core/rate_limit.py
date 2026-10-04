"""Per-endpoint rate limiting, expressed as a FastAPI dependency.

Why ``limits`` and not ``slowapi``
----------------------------------
``slowapi`` was the package pencilled into ``requirements.txt`` for this phase,
and it was evaluated first. Two things ruled it out, both checked rather than
assumed:

* its rate limit is applied as a **decorator that requires a ``request:
  Request`` parameter on every handler it decorates** — it raises
  ``No "request" or "websocket" argument on function`` otherwise. Every
  rate-limited route would have to take a parameter it does not use, which is
  exactly the thickening of routers this codebase has avoided since Phase 2;
* its exception handler produces its own response shape, which would have to be
  replaced anyway to keep the application's single error envelope.

``limits`` is the library ``slowapi`` itself wraps — the window algorithms and
storage backends are the same code. Depending on it directly keeps the
algorithm in a well-supported library, drops one package, and lets the limit be
what this project wanted it to be: an ordinary dependency.

What this does and does not protect
-----------------------------------
Rate limiting here is **abuse and cost control**. It is not an anonymity
mechanism and must never be described as one: it neither hides who is calling
nor prevents a determined caller from spreading requests across addresses.

Nor is it a defence against case-code guessing in any meaningful sense — a case
code carries 100 bits of entropy, so guessing was already hopeless. What it
does is stop one client from flooding the service, cheaply.

Identity, and the reporter's IP address
---------------------------------------
Limits are counted per client address, because that is the only signal an
anonymous endpoint has. The address is **never stored and never logged**: it is
hashed with a salt generated fresh in each process, and only that digest becomes
the counter key. A memory dump of the limiter yields opaque digests, the mapping
dies with the process, and nothing about it reaches the database — ``reports``
has no column for an address and this module adds none.

Known limitations, stated plainly
---------------------------------
* **In-process storage.** Counters live in this worker's memory. Run four
  workers and the effective limit is four times the configured one; restart and
  the window resets. A shared Redis backend fixes both and is a deployment
  concern, not a code change — ``limits`` already speaks Redis.
* **Shared addresses.** Everyone behind one NAT or VPN exit shares a counter.
  For a whistleblowing service this is a real cost: the reporters most likely to
  use Tor or a VPN are the ones most likely to be throttled by someone else's
  traffic. The limits below are set loosely enough that ordinary use is not
  affected.
* **Proxy headers.** ``X-Forwarded-For`` is only believed when
  ``TRUST_PROXY_HEADERS`` is on, because a client can otherwise set it freely
  and mint unlimited identities.
"""

import hashlib
import logging
import secrets
from collections.abc import Callable, Sequence

from fastapi import Depends, Request
from limits import RateLimitItem, parse_many
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter

from app.core.config import Settings, get_settings
from app.core.errors import RateLimitExceededError

logger = logging.getLogger(__name__)

# One salt per process, never written anywhere. Its only job is to make the
# stored counter keys irreversible: without it, a digest of an IPv4 address is
# trivially reversed by enumerating the whole address space.
_KEY_SALT = secrets.token_bytes(32)

# Shared in-memory store. A moving window rather than a fixed one so that a
# caller cannot send a full allowance at 11:59:59 and another at 12:00:00.
_storage = MemoryStorage()
_limiter = MovingWindowRateLimiter(_storage)


def client_fingerprint(request: Request) -> str:
    """An opaque, per-process identifier for the caller.

    Derived from the client address and discarded with the process. The address
    itself is never returned, stored or logged — only this digest, and only in
    memory.
    """
    address = _client_address(request)
    digest = hashlib.sha256(_KEY_SALT + address.encode("utf-8")).hexdigest()
    return digest[:32]


def _client_address(request: Request) -> str:
    """The caller's address, from the proxy header only when it is trusted."""
    settings = _settings_of(request)

    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Left-most entry is the original client; the rest are proxies.
            return forwarded.split(",")[0].strip()

    client = request.client
    return client.host if client else "unknown"


def _settings_of(request: Request) -> Settings:
    """Settings for this request, honouring a test's dependency override."""
    override = request.app.dependency_overrides.get(get_settings)
    return override() if override else get_settings()


class RateLimit:
    """A reusable rate-limit dependency.

    Used as ``Depends(RateLimit(lambda s: s.rate_limit_login, "login"))`` on the
    route. The limit is read from settings at request time rather than captured
    at import, so a test can supply its own without rebuilding the module.

    Nothing about the service layer changes: the dependency either returns or
    raises, and a service is never reached when it raises.
    """

    def __init__(
        self,
        select_limit: Callable[[Settings], str],
        scope: str,
    ) -> None:
        self._select_limit = select_limit
        self._scope = scope

    def __call__(self, request: Request) -> None:
        settings = _settings_of(request)

        if not settings.rate_limit_enabled:
            return

        items = _parse(self._select_limit(settings))
        if not items:
            return

        key = client_fingerprint(request)

        for item in items:
            if _limiter.hit(item, self._scope, key):
                continue

            retry_after = _retry_after_seconds(item, self._scope, key)
            # Logged without the address and without the digest: knowing that
            # *some* caller was throttled on an endpoint is enough to operate
            # the service, and the digest is the one thing that could be
            # correlated across log lines.
            logger.info("Rate limit reached on %s.", self._scope)
            raise RateLimitExceededError(retry_after_seconds=retry_after)


def _parse(expression: str) -> Sequence[RateLimitItem]:
    """Turn ``"5/minute;30/hour"`` into limit items.

    A malformed expression is a configuration error. It is logged and treated
    as *no limit* rather than as a hard failure: an unparseable string must not
    take the reporting endpoint offline, and the log says so loudly.
    """
    try:
        return parse_many(expression)
    except ValueError:
        logger.error("Ignoring an unparseable rate-limit expression: %r", expression)
        return []


def _retry_after_seconds(item: RateLimitItem, *identifiers: str) -> int:
    """How long until the caller may try again, rounded up to a whole second."""
    import time

    stats = _limiter.get_window_stats(item, *identifiers)
    return max(1, int(stats.reset_time - time.time()) + 1)


def reset() -> None:
    """Forget every counter. For tests only."""
    _storage.reset()


# --- The dependencies the routers declare ----------------------------------

ReportSubmissionRateLimit = Depends(
    RateLimit(lambda settings: settings.rate_limit_reports, "reports:submit")
)
CaseLookupRateLimit = Depends(
    RateLimit(lambda settings: settings.rate_limit_case_lookup, "cases:lookup")
)
LoginRateLimit = Depends(RateLimit(lambda settings: settings.rate_limit_login, "auth:login"))

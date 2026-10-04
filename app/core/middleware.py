"""HTTP-level hardening: response headers and a request-body ceiling.

Ordering
--------
Starlette runs the **last-added** middleware outermost, so
:func:`register_middleware` adds them innermost-first. The resulting order for
an incoming request is::

    SecurityHeadersMiddleware   (outermost — stamps every response, errors too)
      └─ CORSMiddleware         (answers preflight, adds CORS to error responses)
          └─ BodySizeLimitMiddleware  (rejects oversized bodies before routing)
              └─ router → rate-limit dependency → service

Security headers sit outermost deliberately: a response produced by CORS's
preflight short-circuit, or by the body-size rejection, still gets them. The
body-size limit sits innermost of the three because it must see the raw request
stream, and because a request rejected there should still carry CORS headers so
a browser reports a useful error rather than an opaque network failure.

None of these read or buffer the request body. The body-size middleware counts
bytes as they stream past and aborts; it never accumulates them.
"""

import logging

from fastapi import FastAPI
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import Settings

logger = logging.getLogger(__name__)

# Paths that render HTML and therefore need a policy that permits the Swagger
# assets. Everything else gets a policy that permits nothing at all.
_DOCS_PATHS = frozenset({"/docs", "/redoc", "/docs/oauth2-redirect"})

# For JSON endpoints: deny everything. An API response is never a document, so
# there is nothing legitimate for a browser to load, frame or submit from it.
_API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"

# Swagger UI is served from a CDN by FastAPI, so the docs pages need script,
# style, image and font sources. Still no framing, and still no form posts to
# anywhere but this origin.
_DOCS_CSP = (
    "default-src 'none'; "
    "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "img-src 'self' https://fastapi.tiangolo.com data:; "
    "font-src 'self' https://cdn.jsdelivr.net; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)

_PERMISSIONS_POLICY = (
    "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
    "magnetometer=(), microphone=(), payment=(), usb=()"
)

#: The headers that do not depend on the path or on configuration.
#:
#: Exported because one response escapes this middleware entirely. Starlette
#: builds its stack as ``ServerErrorMiddleware -> user middleware ->
#: ExceptionMiddleware``, so the 500 produced for an *unhandled* exception is
#: generated outside everything added by :func:`register_middleware`. The
#: handler in ``app/core/errors.py`` therefore applies these itself — and a 500
#: is precisely the response most worth probing, so it must not be the one
#: without hardening headers.
BASE_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": _API_CSP,
    "Permissions-Policy": _PERMISSIONS_POLICY,
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    # Does not name the stack. Note the limitation, which was verified on a
    # real server rather than assumed: uvicorn appends its own
    # ``server: uvicorn`` at the protocol layer, *below* ASGI, so setting this
    # from here results in two Server headers and a client reading the first
    # sees uvicorn's. There is no ASGI hook that can remove it. Run uvicorn
    # with ``--no-server-header`` and this becomes the only one — confirmed.
    "Server": "whistledrop",
}


class SecurityHeadersMiddleware:
    """Stamp hardening headers onto every response.

    Each header below is here for a reason that applies to *this* service.
    Headers that only matter to applications rendering their own HTML are not
    included merely because they are well known.
    """

    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self.app = app
        self._hsts_max_age = settings.hsts_max_age_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                self._apply(headers, path)
            await send(message)

        await self.app(scope, receive, send_with_headers)

    def _apply(self, headers: MutableHeaders, path: str) -> None:
        # Each of these is explained where it is defined, above:
        #   nosniff          — stop a browser second-guessing our Content-Type
        #   no-referrer      — a /docs link must not carry this URL elsewhere
        #   DENY / CSP       — /docs is a real page, so clickjacking applies
        #   Permissions      — this API needs no device access at all
        #   COOP / CORP      — isolate any browsing context from this origin
        #   Server           — do not name the stack for CVE matching
        for header, value in BASE_SECURITY_HEADERS.items():
            headers[header] = value

        # The docs pages load Swagger from a CDN, so they need a policy the
        # strict API one would blank.
        if path in _DOCS_PATHS:
            headers["Content-Security-Policy"] = _DOCS_CSP

        # HSTS only when the deployment actually terminates TLS. Sent over
        # plain HTTP it does nothing at all — browsers ignore it — so shipping
        # it in development would be decoration that invites the false belief
        # that local traffic is protected. It is opt-in via HSTS_MAX_AGE_SECONDS.
        if self._hsts_max_age > 0:
            headers["Strict-Transport-Security"] = (
                f"max-age={self._hsts_max_age}; includeSubDomains"
            )


class BodySizeLimitMiddleware:
    """Refuse request bodies larger than the configured maximum.

    Two checks, in order of cheapness:

    1. ``Content-Length``, when present, is rejected before a single byte of
       body is read.
    2. Otherwise — a chunked upload declares no length — bytes are counted as
       they arrive and the request is failed the moment the total passes the
       limit.

    The second matters: without it, a client could stream gigabytes with no
    ``Content-Length`` and the body would be assembled in memory before any
    application-level length check could look at it.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self._max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                # A non-numeric Content-Length is malformed; let the server
                # layer deal with it rather than guessing what was meant.
                pass

        received = 0
        too_large = False

        async def counting_receive() -> Message:
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes:
                    too_large = True
                    # Present the stream as ended so nothing downstream keeps
                    # buffering; the rejection below is what the client sees.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        sent_response = False

        async def guarded_send(message: Message) -> None:
            nonlocal sent_response
            if too_large and not sent_response:
                sent_response = True
                await self._reject(send)
                return
            if not too_large:
                await send(message)

        await self.app(scope, counting_receive, guarded_send)

        if too_large and not sent_response:
            await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        """Answer with the application's own error envelope."""
        import json

        body = json.dumps(
            {
                "error": {
                    "code": "REQUEST_TOO_LARGE",
                    "message": "The request body is too large.",
                }
            }
        ).encode()

        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def register_middleware(app: FastAPI, settings: Settings) -> None:
    """Install the HTTP-level hardening. Called once from the app factory.

    Added innermost-first; see the module docstring for the resulting order.
    """
    # Innermost of the three.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_body_bytes)

    # CORS is installed only when origins are actually configured. With none —
    # the default, since this API has no browser front end — no CORS headers are
    # emitted at all, which is the most restrictive answer a browser can get.
    if settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allowed_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
            max_age=600,
        )

    # Outermost: every response, including those produced by the two above.
    app.add_middleware(SecurityHeadersMiddleware, settings=settings)

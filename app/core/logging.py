"""Logging setup, and the policy the rest of the code is written against.

What must never be logged
-------------------------
Logs are a lower-trust store than the database: they are shipped to
aggregators, read by more people, retained longer and backed up more casually.
For this service the following must never reach a log line, at any level:

* a plaintext case code, or its ``case_code_hash``;
* the text of a report, or its evidence URL — a URL can itself carry a token;
* a moderator password, a password hash, or any part of one;
* an access token, ``JWT_SECRET_KEY``, or ``CASE_CODE_PEPPER``;
* a reporter's IP address, or anything else identifying a reporter.

The application achieves this by logging *events*, not *values*: every call
site in ``app/`` logs a fixed message plus, at most, a status enum or a request
path. No call site interpolates user input.

Two specific measures back that up
----------------------------------
* The SQLAlchemy engine is created with ``hide_parameters=True``. Without it a
  failed ``INSERT`` raises an error whose text embeds the bound parameters —
  the report body and the case-code hash among them — and ``logger.exception``
  would write the lot to disk. Verified: with the flag off the description
  appears in the exception string; with it on it does not.
* SQL echo is off unconditionally, in every environment. Echoing statements
  would print report text at INFO level.

What is still observable, stated honestly
-----------------------------------------
* **Uvicorn's access log** records method, path, status and client address. It
  is the one place an IP appears, it is written by the server rather than by
  this application, and it is disabled with ``--no-access-log``. Note what it
  cannot capture: case codes travel in request *bodies*, never in URLs — the
  reason ``/cases/lookup`` is a POST — so no access log line has ever contained
  one.
* **Report ids** appear in moderation paths and therefore in access logs. They
  are internal identifiers that say nothing about who filed a report.
* Logs are not an anonymity mechanism and are not claimed to be one. Anyone
  with host access can observe traffic regardless of what is written here.
"""

import logging
import sys

from app.core.config import Settings

#: Loggers that are noisy or that would defeat the policy above if left alone.
_THIRD_PARTY_LEVELS = {
    # INFO here prints every statement's dialect chatter; ERROR keeps genuine
    # failures without narrating queries.
    "sqlalchemy.engine": logging.WARNING,
    "sqlalchemy.pool": logging.WARNING,
    "alembic": logging.INFO,
    # Uvicorn's own access log. Left at its default level; see the module
    # docstring for what it records and how to turn it off.
    "uvicorn.access": logging.INFO,
}

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(settings: Settings) -> None:
    """Install a single stderr handler and quieten the noisy libraries.

    Idempotent: calling it twice does not double every line, which matters
    because the application factory may be invoked more than once in a test
    session.
    """
    level = _resolve_level(settings.log_level)

    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        if getattr(handler, "_whistledrop", False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler._whistledrop = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    for name, third_party_level in _THIRD_PARTY_LEVELS.items():
        logging.getLogger(name).setLevel(third_party_level)

    logging.getLogger("app").setLevel(level)


def _resolve_level(configured: str) -> int:
    """Map a configured level name to a logging level, defaulting to INFO."""
    resolved = logging.getLevelNamesMapping().get(configured.strip().upper())
    return resolved if isinstance(resolved, int) else logging.INFO

"""Engine, session factory and the FastAPI database dependency.

The engine and session factory are built lazily and cached rather than created
at import time. That keeps ``import app.db.session`` free of side effects: the
test suite, Alembic and tooling can import the models without a database
connection being configured behind their back.
"""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings


def create_db_engine(settings: Settings) -> Engine:
    """Build an :class:`Engine` for the given settings.

    Exposed separately from :func:`get_engine` so that tests — and, later, any
    management command — can build an engine against a different DSN without
    touching the cached application engine.
    """
    return create_engine(
        settings.database_url,
        # Connections can be silently killed by a container restart or an idle
        # timeout; pre-ping trades one cheap round trip for not serving an error
        # from a dead connection.
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        # Echoing SQL would put report text into the logs, so it stays off even
        # in debug builds. Use a database-side statement log if it is needed.
        echo=False,
        # Keep bound parameters out of exception messages. Without this, a
        # failed INSERT raises an error whose text embeds the values it tried to
        # write — the report body and the case-code hash among them — and the
        # unhandled-error handler would log the lot. Verified: with this off,
        # the description appears in the exception string; with it on, it does
        # not. The statement itself is still logged, which is what is useful.
        hide_parameters=True,
        future=True,
    )


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide application engine."""
    return create_db_engine(get_settings())


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    return sessionmaker(
        bind=get_engine(),
        class_=Session,
        # Flush only when we say so, so a half-built object cannot reach the
        # database because some unrelated query triggered an autoflush.
        autoflush=False,
        autocommit=False,
        # Attributes stay readable after commit, which lets a request handler
        # return an ORM object it has just committed.
        expire_on_commit=False,
    )


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session.

    The session is closed on the way out whatever happened, and any transaction
    still open when a handler raises is rolled back rather than left to the
    connection pool.
    """
    session = get_sessionmaker()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

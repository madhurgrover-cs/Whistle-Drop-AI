"""Alembic runtime environment.

Two things matter here:

* the DSN is never read from ``alembic.ini`` — it comes from the application's
  Pydantic settings, or from an explicit override, so credentials stay out of
  tracked files;
* ``target_metadata`` is the application's own ``Base.metadata``, which is what
  makes ``alembic revision --autogenerate`` and ``alembic check`` meaningful.
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings

# Importing the model registry is what populates Base.metadata. Without it
# autogenerate would see no tables and cheerfully propose dropping them all.
from app.models import Base

config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers=False is essential, not cosmetic. fileConfig
    # defaults to True, which sets .disabled on every logger not named in
    # alembic.ini — including the whole "app.*" tree. Any process that runs a
    # migration in-process and then serves traffic (a migrate-then-start
    # entrypoint, or the test suite) would afterwards emit no application logs
    # at all, silently. Found during the Phase 5 logging review.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def get_database_url() -> str:
    """Resolve the DSN to migrate, most explicit source first.

    1. ``-x db_url=...`` on the command line.
    2. ``ALEMBIC_DATABASE_URL`` in the environment — how the test suite points
       migrations at the disposable test container.
    3. ``DATABASE_URL`` via the application settings.
    """
    cli_override = context.get_x_argument(as_dictionary=True).get("db_url")
    if cli_override:
        return cli_override

    env_override = os.getenv("ALEMBIC_DATABASE_URL")
    if env_override:
        return env_override

    return get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (``alembic upgrade --sql``)."""
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and run migrations against the resolved database."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Without these, a changed column type or server default is
            # invisible to autogenerate and the schema silently drifts.
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

"""Application configuration.

Every environment-specific or secret value is read from the environment (or a
local ``.env`` file) rather than being written into the source tree. This keeps
credentials out of version control and lets the same image run in development,
test and production with different configuration.

See ``.env.example`` for the full list of supported variables.
"""

from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]


class Settings(BaseSettings):
    """Typed, validated application settings.

    Values are loaded from environment variables first and from ``.env``
    second. Field names map to upper-case variable names, so ``app_name``
    is read from ``APP_NAME``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # docker-compose reads POSTGRES_* from the same .env file; those are not
        # application settings, so unknown keys are ignored rather than fatal.
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "WhistleDrop AI"
    app_env: Environment = "development"
    debug: bool = False
    log_level: str = "INFO"

    api_v1_prefix: str = "/api/v1"

    # The GDG task requires the API to be demonstrable through Swagger, so the
    # interactive docs are enabled by default. A hardened deployment of a real
    # whistleblowing service would gate this behind authentication.
    enable_docs: bool = True

    # --- Database ----------------------------------------------------------
    # Required with no default: the application must fail loudly at startup if
    # it was deployed without a database, rather than silently falling back to
    # something unexpected.
    database_url: str = Field(
        ...,
        description="SQLAlchemy DSN for the application database.",
    )

    # Optional because production has no test database. The integration test
    # suite skips itself with a clear message when this is unset.
    test_database_url: str | None = Field(
        default=None,
        description="SQLAlchemy DSN for the disposable integration-test database.",
    )

    # Connection pool sizing. Exposed so a deployment can tune it without a
    # code change; the defaults suit a single application process.
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)

    # --- Case codes --------------------------------------------------------
    # Server-side key for the HMAC that turns a reporter's case code into the
    # digest stored in the database. Required with no default: a missing pepper
    # must stop the process at startup, because a fallback value would silently
    # make every stored digest forgeable by anyone holding the source code.
    #
    # SecretStr keeps it out of reprs, tracebacks and log lines; reading it
    # takes a deliberate .get_secret_value() call.
    case_code_pepper: SecretStr = Field(
        ...,
        description="HMAC-SHA256 key for case-code hashing. Never logged, never returned.",
    )

    # Minimum pepper length. 32 characters is the SHA-256 block-size floor
    # below which an HMAC key adds no further strength.
    MIN_PEPPER_LENGTH: ClassVar[int] = 32

    # The literal value shipped in .env.example, which is a placeholder and not
    # a key. Rejected outright so it cannot reach a real deployment.
    PLACEHOLDER_PEPPER: ClassVar[str] = "replace-me-with-64-hex-characters-from-the-command-below"

    @field_validator("case_code_pepper")
    @classmethod
    def _pepper_must_be_a_real_key(cls, value: SecretStr) -> SecretStr:
        return _require_real_secret(
            value, "CASE_CODE_PEPPER", cls.PLACEHOLDER_PEPPER, cls.MIN_PEPPER_LENGTH
        )

    # --- Moderator authentication -----------------------------------------
    # Signing key for moderator access tokens. Required with no default, for
    # the same reason as the pepper: a built-in fallback would let anyone
    # holding the source mint a valid moderator token.
    #
    # Rotating this invalidates every token in circulation immediately, which
    # is the intended emergency response to a suspected leak.
    jwt_secret_key: SecretStr = Field(
        ...,
        description="HMAC signing key for moderator access tokens. Never logged, never returned.",
    )

    # HS256: symmetric, one service both signs and verifies, no key
    # distribution problem to solve. Restricted to the HMAC family so a
    # misconfiguration cannot select "none" or an asymmetric algorithm whose
    # public key an attacker could supply.
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"

    # 30 minutes. Short enough that a leaked token has a small window, long
    # enough that a moderator is not re-authenticating mid-review. There is no
    # refresh token yet: expiry means logging in again.
    jwt_access_token_expire_minutes: int = Field(default=30, ge=1, le=1440)

    # Not a credential: the sentinel shipped in .env.example, matched here so
    # it can be refused. S105 flags it as a hardcoded password, which is
    # exactly backwards — this constant exists to stop one being used.
    PLACEHOLDER_JWT_SECRET: ClassVar[str] = "replace-me-with-a-different-64-hex-value"  # noqa: S105
    MIN_JWT_SECRET_LENGTH: ClassVar[int] = 32

    @field_validator("jwt_secret_key")
    @classmethod
    def _jwt_secret_must_be_a_real_key(cls, value: SecretStr) -> SecretStr:
        return _require_real_secret(
            value, "JWT_SECRET_KEY", cls.PLACEHOLDER_JWT_SECRET, cls.MIN_JWT_SECRET_LENGTH
        )

    # --- Hardening ---------------------------------------------------------
    # Rate limiting. On by default; the test suite turns it off except in the
    # tests that exercise it, so that a long suite is not throttled by its own
    # traffic.
    rate_limit_enabled: bool = True

    # Per-endpoint limits, expressed in the `limits` library's notation. Chosen
    # for how these endpoints are actually used by a person, with headroom:
    #
    #   reports  — filing is a rare, deliberate act. A handful a minute is far
    #              beyond real use and still blunts automated flooding.
    #   lookup   — a reporter checks their case occasionally. This also caps
    #              case-code guessing, though 100 bits of entropy already makes
    #              guessing hopeless; the real purpose is cost control.
    #   login    — the tightest. This is the credential-stuffing surface, and a
    #              moderator who cannot type their password five times a minute
    #              has a different problem.
    rate_limit_reports: str = "5/minute;30/hour"
    rate_limit_case_lookup: str = "10/minute;60/hour"
    rate_limit_login: str = "5/minute;20/hour"

    # Whether to believe X-Forwarded-For. Off by default: the header is trivially
    # spoofable, and trusting it unconditionally lets one client mint unlimited
    # rate-limit identities. Turn it on only behind a proxy that overwrites it.
    trust_proxy_headers: bool = False

    # Largest request body accepted, in bytes. 64 KiB comfortably fits the
    # 20,000-character description plus a 2,048-character URL and JSON
    # overhead, while keeping anything larger from being read into memory.
    max_request_body_bytes: int = Field(default=64 * 1024, ge=1024, le=10 * 1024 * 1024)

    # Allowed browser origins. Empty by default — this API has no browser
    # front end, so nothing needs cross-origin access and the safest policy is
    # none at all. Set a comma-separated list only when a front end exists.
    cors_allowed_origins: list[str] = Field(default_factory=list)

    # Whether cross-origin requests may carry credentials. Off, and guarded
    # below: the API authenticates with a bearer header rather than cookies,
    # so there is nothing for the browser to attach.
    cors_allow_credentials: bool = False

    # HSTS max-age in seconds, sent only when the deployment actually terminates
    # TLS. 0 disables the header. It is meaningless over plain HTTP — browsers
    # ignore it — so it stays off outside production.
    hsts_max_age_seconds: int = Field(default=0, ge=0)

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string, since that is what an env var holds."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @model_validator(mode="after")
    def _cors_must_not_be_wildcard_with_credentials(self) -> "Settings":
        """The combination browsers reject and that would be unsafe anyway.

        ``Access-Control-Allow-Origin: *`` with credentials would let any site
        make authenticated cross-origin calls. Refused at startup rather than
        left to be noticed later.
        """
        if self.cors_allow_credentials and "*" in self.cors_allowed_origins:
            raise ValueError(
                "CORS_ALLOWED_ORIGINS must not contain '*' when "
                "CORS_ALLOW_CREDENTIALS is true. Name the origins explicitly."
            )
        return self

    # --- AI-assisted triage ------------------------------------------------
    # Whether to run triage inference after a report is filed. Off turns the
    # feature off cleanly: reports are still filed, and no triage row is
    # written. Nothing about reporting depends on it.
    ml_triage_enabled: bool = True

    # The model version the server will load. Pinned, never "latest", and never
    # selectable by a request — a client that could choose a model version
    # could choose one the operator did not vet.
    ml_model_version: str = "whistledrop-category-v1"

    # Where artifacts live. An operator setting, never a request parameter, so
    # no path from a client can reach joblib.load.
    ml_artifact_root: Path = Path("ml/artifacts")

    # Load the model at startup rather than on the first report.
    #
    # Loading costs roughly three seconds, almost all of it importing
    # scikit-learn beneath joblib. Left lazy, that cost lands on whichever
    # reporter happens to submit first after a restart — the one person who
    # should not be kept waiting. Warming at startup moves it to a moment when
    # nobody is waiting.
    #
    # The loader itself stays lazy; this only triggers it early, and a failure
    # is logged and ignored, so a missing or corrupt artifact still cannot stop
    # the application starting. Off in tests, where thousands of app instances
    # are built and none should pay for it.
    ml_warm_start: bool = True

    # bcrypt work factor. 12 costs roughly a quarter-second per hash on
    # commodity hardware in 2026 — enough to make offline cracking of a stolen
    # hash expensive, little enough that a login feels instant. The test suite
    # drops it to the minimum so that hashing does not dominate the run.
    password_hash_rounds: int = Field(default=12, ge=4, le=16)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @model_validator(mode="after")
    def _production_must_not_use_test_grade_hashing(self) -> "Settings":
        """A low work factor is a test convenience, never a deployment setting."""
        if self.is_production and self.password_hash_rounds < 12:
            raise ValueError(
                "PASSWORD_HASH_ROUNDS must be at least 12 in production; "
                f"got {self.password_hash_rounds}."
            )
        return self


def _require_real_secret(
    value: SecretStr, name: str, placeholder: str, minimum_length: int
) -> SecretStr:
    """Reject the shipped placeholder and anything too short to be a key."""
    secret = value.get_secret_value()
    if secret == placeholder:
        raise ValueError(
            f"{name} is still the placeholder from .env.example. "
            'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
        )
    if len(secret) < minimum_length:
        raise ValueError(f"{name} must be at least {minimum_length} characters.")
    return value


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings.

    Cached so that the ``.env`` file is parsed once per process. Used as a
    FastAPI dependency, which also makes it straightforward to override with
    test settings via ``app.dependency_overrides``.
    """
    return Settings()  # type: ignore[call-arg]  # values come from the environment

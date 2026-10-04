"""Request and response schemas for moderator authentication.

Note what has no schema here: there is no registration request model, because
there is no registration endpoint. Moderator accounts exist only through the
seeding CLI.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.core.passwords import MAX_PASSWORD_BYTES

# Generous bounds only. Login must not enforce the *policy* — rejecting a short
# password at the schema would tell an attacker their guess was too short to be
# anyone's real password, and would make a policy change retroactively lock
# people out. The only job here is to bound the input before it reaches bcrypt.
USERNAME_MAX_LENGTH = 64
PASSWORD_MAX_LENGTH = MAX_PASSWORD_BYTES


class LoginRequest(BaseModel):
    """Moderator credentials."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"username": "alex", "password": "correct-horse-battery"}]},
    )

    username: Annotated[
        str,
        Field(
            min_length=1,
            max_length=USERNAME_MAX_LENGTH,
            description="Your moderator username. Case and surrounding spaces are ignored.",
        ),
    ]

    # SecretStr so the value cannot reach a log line, a traceback frame or a
    # repr by accident. Reading it takes a deliberate .get_secret_value().
    password: Annotated[
        SecretStr,
        Field(
            min_length=1,
            max_length=PASSWORD_MAX_LENGTH,
            description="Your password. Never stored, never logged, never returned.",
        ),
    ]


class TokenResponse(BaseModel):
    """A newly issued access token.

    Contains nothing about the account beyond the token itself: no id, no
    username, no ``is_active``, no timestamps. A client that needs those will
    get them from a Phase 4 endpoint behind the token, not from the login
    response.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIu...",
                    "token_type": "bearer",
                    "expires_in": 1800,
                }
            ]
        }
    )

    access_token: str = Field(description="Signed JWT. Send it as `Authorization: Bearer <token>`.")
    token_type: str = Field(
        default="bearer",
        description="Always `bearer`, per RFC 6750.",
    )
    expires_in: int = Field(
        description="Seconds until the token expires. After that, log in again.",
    )

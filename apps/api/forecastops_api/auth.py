"""Authenticate a caller and read the tenant the request may access.

Cloud mode requires a Cognito access token. The tenant id is the ``tenant_id``
claim. Local mode skips this check when authentication is disabled and uses
the ``local`` tenant.
"""

from dataclasses import dataclass
from typing import Protocol

import jwt
from jwt import PyJWKClient

from forecastops_api.errors import ApiError
from forecastops_api.settings import Settings
from forecastops_api.tenancy import LOCAL_TENANT, validate_tenant_id


@dataclass(frozen=True)
class Caller:
    """The tenant and subject allowed to read and write on this request."""

    tenant_id: str
    subject: str | None = None


class TokenVerifier(Protocol):
    """Validate one bearer token and return the caller it names."""

    def verify(self, token: str) -> Caller:
        """Return the caller, or raise an API error when the token is refused."""


class SigningKeys(Protocol):
    """Resolve the public key that signed a token."""

    def key_for(self, token: str) -> object:
        """Return the key PyJWT should use to check ``token``."""


class JwksSigningKeys:
    """Load Cognito JSON Web Keys for one user pool."""

    def __init__(self, issuer: str) -> None:
        self._client = PyJWKClient(f"{issuer}/.well-known/jwks.json")

    def key_for(self, token: str) -> object:
        """Return the public key advertised for ``token``."""

        return self._client.get_signing_key_from_jwt(token).key


class CognitoTokenVerifier:
    """Check a Cognito access token and require a ``tenant_id`` claim."""

    def __init__(self, *, issuer: str, client_id: str, keys: SigningKeys) -> None:
        self._issuer = issuer
        self._client_id = client_id
        self._keys = keys

    def verify(self, token: str) -> Caller:
        """Return the caller named by ``token``."""

        payload = _decode_access_token(
            token,
            issuer=self._issuer,
            client_id=self._client_id,
            keys=self._keys,
        )
        tenant = payload.get("tenant_id")
        if not isinstance(tenant, str) or tenant.strip() == "":
            raise ApiError(403, "forbidden", "The token is missing a tenant.")
        try:
            tenant_id = validate_tenant_id(tenant)
        except ValueError as exc:
            raise ApiError(403, "forbidden", "The token tenant is not valid.") from exc
        subject = payload.get("sub")
        return Caller(
            tenant_id=tenant_id,
            subject=subject if isinstance(subject, str) else None,
        )


def authentication_required(settings: Settings) -> bool:
    """Require a bearer token when authentication is enabled.

    Cloud settings cannot be loaded with authentication disabled. Local mode
    leaves the flag off and uses the ``local`` tenant.
    """

    return settings.auth_enabled


def local_caller() -> Caller:
    """Return the tenant used when local mode leaves authentication off."""

    return Caller(tenant_id=LOCAL_TENANT, subject=None)


def build_token_verifier(settings: Settings) -> CognitoTokenVerifier:
    """Build the Cognito verifier for these settings.

    The key set is fetched on the first token, not at process start.
    """

    region = settings.aws_region.strip() or "us-east-1"
    pool_id = settings.cognito_user_pool_id.strip()
    issuer = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"
    return CognitoTokenVerifier(
        issuer=issuer,
        client_id=settings.cognito_app_client_id.strip(),
        keys=JwksSigningKeys(issuer),
    )


def _decode_access_token(
    token: str,
    *,
    issuer: str,
    client_id: str,
    keys: SigningKeys,
) -> dict[str, object]:
    try:
        key = keys.key_for(token)
        payload = jwt.decode(
            token,
            key,  # type: ignore[arg-type]
            algorithms=["RS256"],
            issuer=issuer,
            options={"verify_aud": False},
        )
    except (jwt.PyJWTError, ValueError) as exc:
        raise ApiError(401, "unauthorized", "The bearer token is invalid.") from exc
    if not isinstance(payload, dict):
        raise ApiError(401, "unauthorized", "The bearer token is invalid.")
    if payload.get("token_use") != "access" or payload.get("client_id") != client_id:
        raise ApiError(401, "unauthorized", "The bearer token is invalid.")
    return {str(key): value for key, value in payload.items()}

"""Artifact storage for dataset, model, and forecast objects.

Services depend on :class:`ArtifactStore`. Settings choose a directory on disk
or one remote bucket. Object bodies are never written to logs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Protocol, cast, runtime_checkable

from forecastops_api.settings import Settings
from forecastops_api.tenancy import LOCAL_TENANT, validate_tenant_id

ALLOWED_PREFIXES: tuple[str, ...] = (
    "raw",
    "processed",
    "features",
    "training",
    "models",
    "forecasts",
    "evaluations",
)
PRESIGNED_UPLOAD_SECONDS = 900


class InvalidArtifactPrefix(ValueError):
    """Raised when a key is outside the allowed top-level prefixes."""


class ArtifactNotFound(FileNotFoundError):
    """Raised when no object exists at a uri."""


class ArtifactStore(Protocol):
    """Read and write artifact bytes addressed by prefix."""

    def for_tenant(self, tenant_id: str) -> ArtifactStore:
        """Return a store that refuses keys outside ``tenant_id``."""

    def put(self, prefix: str, name: str, body: bytes) -> str:
        """Store ``body`` and return its uri."""

    def get(self, uri: str) -> bytes:
        """Return the bytes stored at ``uri``."""

    def list(self, prefix: str) -> list[str]:
        """Return uris stored under ``prefix``."""

    def delete(self, uri: str) -> None:
        """Remove the object at ``uri`` if it exists."""


@runtime_checkable
class PresigningStore(Protocol):
    """Issue a short-lived upload URL for one object."""

    def presign_put(
        self,
        prefix: str,
        name: str,
        *,
        expires_in: int = PRESIGNED_UPLOAD_SECONDS,
    ) -> tuple[str, str]:
        """Return a pre-signed upload URL and the object uri."""


class ObjectStorageClient(Protocol):
    """Subset of the remote object API used by :class:`S3ArtifactStore`."""

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> Mapping[str, object]:
        """Write one object."""

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, object]:
        """Read one object."""

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str,
        ContinuationToken: str | None = None,
    ) -> Mapping[str, object]:
        """List object keys under ``Prefix``."""

    def delete_object(self, *, Bucket: str, Key: str) -> Mapping[str, object]:
        """Delete one object."""

    def generate_presigned_url(
        self,
        ClientMethod: str,
        *,
        Params: Mapping[str, str],
        ExpiresIn: int,
    ) -> str:
        """Return a pre-signed URL for ``ClientMethod``."""


def normalize_prefix(prefix: str) -> str:
    """Return a top-level prefix, or reject anything outside the allowed list."""

    cleaned = prefix.strip().strip("/")
    if cleaned not in ALLOWED_PREFIXES:
        allowed = ", ".join(f"{name}/" for name in ALLOWED_PREFIXES)
        raise InvalidArtifactPrefix(f"Artifact prefix must be one of: {allowed}.")
    return cleaned


def tenant_key_prefix(tenant_id: str) -> str:
    """Return the object-key prefix reserved for ``tenant_id``."""

    return f"tenant/{_tenant_or_reject(tenant_id)}"


def require_allowed_key(key: str, *, tenant_id: str | None = None) -> str:
    """Return ``key`` when it stays inside one tenant and an allowed prefix."""

    cleaned = key.strip().lstrip("/")
    if "\\" in cleaned or cleaned.startswith("/") or "//" in key:
        raise InvalidArtifactPrefix("Artifact key must stay inside the tenant prefix.")
    parts = cleaned.split("/")
    if len(parts) < 4 or parts[0] != "tenant":
        raise InvalidArtifactPrefix(
            "Artifact key must start with tenant/{tenant_id}/ and an allowed prefix."
        )
    owner = _tenant_or_reject(parts[1])
    if tenant_id is not None and owner != _tenant_or_reject(tenant_id):
        raise InvalidArtifactPrefix("Artifact key is outside this tenant.")
    normalized = normalize_prefix(parts[2])
    name = "/".join(parts[3:])
    _reject_unsafe_name(name)
    return f"tenant/{owner}/{normalized}/{name}"


def object_key(prefix: str, name: str, *, tenant_id: str = LOCAL_TENANT) -> str:
    """Join a tenant, an allowed prefix, and a relative object name."""

    normalized = normalize_prefix(prefix)
    _reject_unsafe_name(name)
    return f"{tenant_key_prefix(tenant_id)}/{normalized}/{name.lstrip('/')}"


def select_artifact_store(
    settings: Settings,
    *,
    client: ObjectStorageClient | None = None,
) -> ArtifactStore:
    """Use the local directory unless AWS is enabled."""

    if not settings.aws_enabled:
        return LocalArtifactStore(settings.artifact_dir)
    bucket = settings.artifacts_bucket.strip()
    if not bucket:
        raise ValueError("ARTIFACTS_BUCKET is required when AWS is enabled.")
    resolved = client if client is not None else build_s3_client(settings.aws_region)
    return S3ArtifactStore(bucket, resolved)


def build_s3_client(region: str) -> ObjectStorageClient:
    """Construct the remote object client.

    The SDK import stays inside this function so local mode never loads it.
    """

    import boto3

    return cast(ObjectStorageClient, boto3.client("s3", region_name=region))


class LocalArtifactStore:
    """Store artifacts as files under one directory."""

    def __init__(self, root: Path, *, tenant_id: str = LOCAL_TENANT) -> None:
        self._root = root.expanduser().resolve()
        self._tenant_id = validate_tenant_id(tenant_id)

    def for_tenant(self, tenant_id: str) -> LocalArtifactStore:
        """Return a store bound to ``tenant_id`` in the same directory."""

        return LocalArtifactStore(self._root, tenant_id=tenant_id)

    def put(self, prefix: str, name: str, body: bytes) -> str:
        """Write ``body`` under ``prefix`` and return its path."""

        path = self._path_for_key(object_key(prefix, name, tenant_id=self._tenant_id))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return str(path)

    def get(self, uri: str) -> bytes:
        """Read the file at ``uri``."""

        path = self._path_for_uri(uri)
        if not path.is_file():
            raise ArtifactNotFound(uri)
        return path.read_bytes()

    def list(self, prefix: str) -> list[str]:
        """Return file paths stored under ``prefix``."""

        directory = self._root / "tenant" / self._tenant_id / normalize_prefix(prefix)
        if not directory.exists():
            return []
        return sorted(str(path.resolve()) for path in directory.rglob("*") if path.is_file())

    def delete(self, uri: str) -> None:
        """Remove the file at ``uri`` when it exists."""

        path = self._path_for_uri(uri)
        if path.is_dir():
            raise InvalidArtifactPrefix("Artifact uri must name an object.")
        path.unlink(missing_ok=True)

    def _path_for_key(self, key: str) -> Path:
        relative = Path(*require_allowed_key(key, tenant_id=self._tenant_id).split("/"))
        return self._root / relative

    def _path_for_uri(self, uri: str) -> Path:
        candidate = Path(uri)
        if not candidate.is_absolute():
            candidate = self._root / uri
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(self._root)
        except ValueError as exc:
            raise InvalidArtifactPrefix("Artifact uri is outside the store.") from exc
        require_allowed_key(relative.as_posix(), tenant_id=self._tenant_id)
        return resolved


class S3ArtifactStore:
    """Store artifacts in one remote bucket."""

    def __init__(
        self,
        bucket: str,
        client: ObjectStorageClient,
        *,
        tenant_id: str = LOCAL_TENANT,
    ) -> None:
        cleaned = bucket.strip()
        if not cleaned:
            raise ValueError("Artifact bucket name is required.")
        self._bucket = cleaned
        self._client = client
        self._tenant_id = validate_tenant_id(tenant_id)

    def for_tenant(self, tenant_id: str) -> S3ArtifactStore:
        """Return a store bound to ``tenant_id`` in the same bucket."""

        return S3ArtifactStore(self._bucket, self._client, tenant_id=tenant_id)

    def put(self, prefix: str, name: str, body: bytes) -> str:
        """Upload ``body`` and return its object uri."""

        key = object_key(prefix, name, tenant_id=self._tenant_id)
        self._client.put_object(Bucket=self._bucket, Key=key, Body=body)
        return self._uri(key)

    def get(self, uri: str) -> bytes:
        """Download the object at ``uri``."""

        key = self._key_for_uri(uri)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if _object_missing(exc):
                raise ArtifactNotFound(uri) from exc
            raise
        payload = response.get("Body")
        read = getattr(payload, "read", None)
        if not callable(read):
            raise ArtifactNotFound(uri)
        body = read()
        if not isinstance(body, bytes):
            raise ArtifactNotFound(uri)
        return body

    def list(self, prefix: str) -> list[str]:
        """Return object uris stored under ``prefix``."""

        key_prefix = f"{tenant_key_prefix(self._tenant_id)}/{normalize_prefix(prefix)}/"
        uris: list[str] = []
        token: str | None = None
        seen: set[str] = set()
        while True:
            page = self._list_page(key_prefix, token)
            contents = page.get("Contents", [])
            if isinstance(contents, list):
                for item in contents:
                    if not isinstance(item, Mapping):
                        continue
                    key = item.get("Key")
                    if isinstance(key, str):
                        uris.append(self._uri(require_allowed_key(key, tenant_id=self._tenant_id)))
            if page.get("IsTruncated") is not True:
                break
            next_token = page.get("NextContinuationToken")
            if not isinstance(next_token, str) or next_token in seen:
                break
            seen.add(next_token)
            token = next_token
        return sorted(uris)

    def delete(self, uri: str) -> None:
        """Delete the object at ``uri``."""

        key = self._key_for_uri(uri)
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def presign_put(
        self,
        prefix: str,
        name: str,
        *,
        expires_in: int = PRESIGNED_UPLOAD_SECONDS,
    ) -> tuple[str, str]:
        """Return a pre-signed upload URL and the object uri."""

        key = object_key(prefix, name, tenant_id=self._tenant_id)
        url = self._client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        return url, self._uri(key)

    def _list_page(self, key_prefix: str, token: str | None) -> Mapping[str, object]:
        if token is None:
            return self._client.list_objects_v2(Bucket=self._bucket, Prefix=key_prefix)
        return self._client.list_objects_v2(
            Bucket=self._bucket,
            Prefix=key_prefix,
            ContinuationToken=token,
        )

    def _uri(self, key: str) -> str:
        return f"s3://{self._bucket}/{key}"

    def _key_for_uri(self, uri: str) -> str:
        marker = "s3://"
        if not uri.startswith(marker):
            raise InvalidArtifactPrefix("Artifact uri must use the s3:// scheme.")
        rest = uri[len(marker) :]
        bucket, separator, key = rest.partition("/")
        if separator == "" or bucket != self._bucket or key == "":
            raise InvalidArtifactPrefix("Artifact uri is outside this bucket.")
        return require_allowed_key(key, tenant_id=self._tenant_id)


def upload_object_name(filename: str) -> str:
    """Keep only the final path segment of an uploaded file name."""

    candidate = filename.replace("\\", "/").split("/")[-1].strip()
    if candidate in {"", ".", ".."}:
        raise InvalidArtifactPrefix("Dataset file name is required.")
    object_key("raw", candidate, tenant_id=LOCAL_TENANT)
    return candidate


def _tenant_or_reject(value: str) -> str:
    try:
        return validate_tenant_id(value)
    except ValueError as exc:
        raise InvalidArtifactPrefix(str(exc)) from exc


def _reject_unsafe_name(name: str) -> None:
    if name == "" or name.startswith("/") or "\\" in name:
        raise InvalidArtifactPrefix("Artifact name must be a relative path inside the prefix.")
    parts = PurePosixPath(name).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise InvalidArtifactPrefix("Artifact name must stay inside the prefix.")


def _object_missing(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    return error.get("Code") in {"NoSuchKey", "404", "NotFound"}

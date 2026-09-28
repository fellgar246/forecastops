"""Artifact storage for dataset, model, and forecast objects.

Services depend on :class:`ArtifactStore`. Settings choose a directory on disk
or one remote bucket. Object bodies are never written to logs.
"""

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Protocol, cast, runtime_checkable

from forecastops_api.settings import Settings

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


def require_allowed_key(key: str) -> str:
    """Return ``key`` when its top-level prefix is allowed."""

    cleaned = key.strip().lstrip("/")
    if "/" not in cleaned:
        raise InvalidArtifactPrefix("Artifact key must include an allowed prefix and a name.")
    prefix, name = cleaned.split("/", 1)
    normalized = normalize_prefix(prefix)
    _reject_unsafe_name(name)
    return f"{normalized}/{name}"


def object_key(prefix: str, name: str) -> str:
    """Join an allowed prefix and a relative object name."""

    normalized = normalize_prefix(prefix)
    _reject_unsafe_name(name)
    return f"{normalized}/{name.lstrip('/')}"


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

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()

    def put(self, prefix: str, name: str, body: bytes) -> str:
        """Write ``body`` under ``prefix`` and return its path."""

        path = self._path_for_key(object_key(prefix, name))
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

        directory = self._root / normalize_prefix(prefix)
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
        relative = Path(*require_allowed_key(key).split("/"))
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
        require_allowed_key(relative.as_posix())
        return resolved


class S3ArtifactStore:
    """Store artifacts in one remote bucket."""

    def __init__(self, bucket: str, client: ObjectStorageClient) -> None:
        cleaned = bucket.strip()
        if not cleaned:
            raise ValueError("Artifact bucket name is required.")
        self._bucket = cleaned
        self._client = client

    def put(self, prefix: str, name: str, body: bytes) -> str:
        """Upload ``body`` and return its object uri."""

        key = object_key(prefix, name)
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

        key_prefix = f"{normalize_prefix(prefix)}/"
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
                        uris.append(self._uri(require_allowed_key(key)))
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

        key = object_key(prefix, name)
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
        return require_allowed_key(key)


def upload_object_name(filename: str) -> str:
    """Keep only the final path segment of an uploaded file name."""

    candidate = filename.replace("\\", "/").split("/")[-1].strip()
    if candidate in {"", ".", ".."}:
        raise InvalidArtifactPrefix("Dataset file name is required.")
    object_key("raw", candidate)
    return candidate


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

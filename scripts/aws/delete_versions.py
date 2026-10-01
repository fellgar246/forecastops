"""Build Dynamo-free S3 delete requests for leftover object versions.

Destroy leaves versioned objects in place. This helper turns a
list-object-versions payload into delete-objects batches of at most 1000.
"""

import json
import sys
from collections.abc import Mapping
from typing import Any


def version_delete_batches(listing: Mapping[str, Any]) -> list[dict[str, object]]:
    """Return delete-objects payloads for versions and delete markers."""

    objects: list[dict[str, str]] = []
    for collection in ("Versions", "DeleteMarkers"):
        entries = listing.get(collection) or []
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, Mapping):
                continue
            key = item.get("Key")
            version = item.get("VersionId")
            if isinstance(key, str) and isinstance(version, str):
                objects.append({"Key": key, "VersionId": version})
    batches: list[dict[str, object]] = []
    for start in range(0, len(objects), 1000):
        batches.append({"Objects": objects[start : start + 1000], "Quiet": True})
    return batches


def main() -> None:
    """Read a listing on stdin and write one JSON batch per line."""

    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise SystemExit("Object version listing must be a JSON object.")
    for batch in version_delete_batches(payload):
        json.dump(batch, sys.stdout)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()

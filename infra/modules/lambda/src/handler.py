"""Health stub for an unpackaged control-plane function.

`make aws-deploy` replaces this file with the HTTP API package. This stub
does not train models or call inference.
"""

import json
from typing import Any


def handler(_event: dict[str, Any], _context: object) -> dict[str, Any]:
    """Return a static health payload."""

    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"status": "healthy"}),
    }

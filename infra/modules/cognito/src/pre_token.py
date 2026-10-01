"""Copy the user's tenant attribute onto the access token.

The API reads the claim ``tenant_id``. A token without it is rejected.
"""

from typing import Any


def handler(event: dict[str, Any], _context: object) -> dict[str, Any]:
    """Add ``tenant_id`` when the user has a tenant attribute."""

    attributes = event.get("request", {}).get("userAttributes", {})
    tenant = attributes.get("custom:tenant_id", "")
    if not isinstance(tenant, str) or tenant.strip() == "":
        return event
    response = event.setdefault("response", {})
    response["claimsAndScopeOverrideDetails"] = {
        "accessTokenGeneration": {
            "claimsToAddOrOverride": {"tenant_id": tenant.strip()},
        }
    }
    return event

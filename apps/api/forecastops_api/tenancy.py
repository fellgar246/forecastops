"""Tenant identity shared by metadata rows and artifact keys."""

import re
from typing import Protocol, cast

LOCAL_TENANT = "local"
_TENANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_tenant_id(value: str) -> str:
    """Return a tenant id that can be used in a key and a database column."""

    cleaned = value.strip()
    if _TENANT_ID.fullmatch(cleaned) is None:
        raise ValueError("Tenant id must be letters, numbers, hyphens, or underscores.")
    return cleaned


class _TenantStamped(Protocol):
    tenant_id: str


def stamp_tenant(row: object, tenant_id: str) -> None:
    """Assign ``tenant_id`` so a write cannot choose another tenant."""

    cast(_TenantStamped, row).tenant_id = validate_tenant_id(tenant_id)


def row_tenant_id(row: object) -> str:
    """Return the row's tenant, treating a missing value as the local tenant."""

    value = getattr(row, "tenant_id", None)
    if isinstance(value, str) and value.strip():
        return value
    return LOCAL_TENANT

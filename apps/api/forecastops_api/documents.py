"""Encode domain rows as JSON documents.

Both metadata adapters store these documents. Dates and timestamps keep their
type tags so a round trip returns the same Python values.
"""

from datetime import UTC, date, datetime
from typing import cast

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Mapper

from forecastops_api.db import Base

_TYPE = "__type__"
_VALUE = "value"


def document_from_row(row: object) -> dict[str, object]:
    """Return the JSON-ready document for ``row``."""

    payload: dict[str, object] = {}
    for column in mapper_for(row).columns:
        payload[column.key] = _encode(getattr(row, column.key))
    encoded = _encode(payload)
    if not isinstance(encoded, dict):
        raise TypeError("Metadata document must be an object.")
    return {str(key): value for key, value in encoded.items()}


def row_from_document[RowT: Base](model: type[RowT], payload: dict[str, object]) -> RowT:
    """Build ``model`` from a stored document."""

    decoded = _decode(payload)
    if not isinstance(decoded, dict):
        raise ValueError("Metadata document must be an object.")
    columns = {column.key for column in sa_inspect(model).columns}
    values = {str(key): value for key, value in decoded.items() if str(key) in columns}
    return model(**values)


def mapper_for(row: object) -> Mapper[Base]:
    """Return the mapper for a domain row."""

    state = sa_inspect(row)
    if state is None:
        raise TypeError("Metadata row cannot be read.")
    return cast(Mapper[Base], state.mapper)


def as_utc(value: datetime) -> datetime:
    """Return ``value`` as an aware UTC timestamp."""

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _encode(value: object) -> object:
    if isinstance(value, datetime):
        return {_TYPE: "datetime", _VALUE: value.isoformat()}
    if isinstance(value, date):
        return {_TYPE: "date", _VALUE: value.isoformat()}
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, float | int | str):
        return value
    raise TypeError(f"Cannot store {type(value).__name__} in a metadata document.")


def _decode(value: object) -> object:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        marker = value.get(_TYPE)
        stored = value.get(_VALUE)
        if marker == "datetime" and isinstance(stored, str) and set(value) == {_TYPE, _VALUE}:
            parsed = datetime.fromisoformat(stored)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed
        if marker == "date" and isinstance(stored, str) and set(value) == {_TYPE, _VALUE}:
            return date.fromisoformat(stored)
        return {str(key): _decode(item) for key, item in value.items()}
    return value

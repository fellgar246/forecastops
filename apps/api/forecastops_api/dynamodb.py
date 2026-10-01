"""DynamoDB adapter for domain documents.

The table key is the row's table name and id. A local double implements the
same put, get, and query operations as the boto3 table wrapper.
"""

import json
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Protocol, cast

from sqlalchemy import Integer, Table

from forecastops_api.db import Base
from forecastops_api.documents import as_utc, document_from_row, mapper_for, row_from_document
from forecastops_api.persistence import (
    AIExplanationRow,
    DatasetRow,
    ForecastErrorEvaluationRow,
    ForecastPointRow,
    ForecastRunRow,
    ModelVersionRow,
    MonitoringReportRow,
    PromotionDecisionRow,
    RetrainRequestRow,
    TrainingRunRow,
)
from forecastops_api.repositories import ExplanationUsageTotals
from forecastops_api.settings import Settings
from forecastops_api.tenancy import LOCAL_TENANT, row_tenant_id, stamp_tenant, validate_tenant_id

_IDEMPOTENCY = "forecast_idempotency"
_META = "__meta__"


class MetadataTable(Protocol):
    """Put, get, and query JSON documents addressed by partition and sort key."""

    def put_item(self, pk: str, sk: str, document: str) -> None:
        """Store one document."""

    def get_item(self, pk: str, sk: str) -> Mapping[str, str] | None:
        """Return one item, or none when the key is absent."""

    def query_partition(self, pk: str) -> list[Mapping[str, str]]:
        """Return every item in partition ``pk``."""


class DynamoItemTable(Protocol):
    """Subset of a boto3 DynamoDB table resource used by the adapter."""

    def put_item(self, *, Item: Mapping[str, str]) -> object:
        """Store one item."""

    def get_item(
        self,
        *,
        Key: Mapping[str, str],
        ConsistentRead: bool = False,
    ) -> Mapping[str, object]:
        """Read one item."""

    def query(self, **kwargs: object) -> Mapping[str, object]:
        """Query one partition."""


class MemoryMetadataTable:
    """In-memory DynamoDB double. Items use the same pk, sk, and document shape."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, str]] = {}

    def put_item(self, pk: str, sk: str, document: str) -> None:
        """Store one document."""

        self.items[(pk, sk)] = {"pk": pk, "sk": sk, "document": document}

    def get_item(self, pk: str, sk: str) -> Mapping[str, str] | None:
        """Return one item, or none when the key is absent."""

        item = self.items.get((pk, sk))
        if item is None:
            return None
        return dict(item)

    def query_partition(self, pk: str) -> list[Mapping[str, str]]:
        """Return every item in partition ``pk``."""

        return [dict(item) for (item_pk, _sk), item in self.items.items() if item_pk == pk]


class BotoMetadataTable:
    """DynamoDB table resource that stores one JSON document per domain row."""

    def __init__(self, table: DynamoItemTable) -> None:
        self._table = table

    def put_item(self, pk: str, sk: str, document: str) -> None:
        """Store one document."""

        self._table.put_item(Item={"pk": pk, "sk": sk, "document": document})

    def get_item(self, pk: str, sk: str) -> Mapping[str, str] | None:
        """Return one item, or none when the key is absent."""

        response = self._table.get_item(Key={"pk": pk, "sk": sk}, ConsistentRead=True)
        item = response.get("Item")
        if not isinstance(item, Mapping) or "document" not in item:
            return None
        return {"pk": pk, "sk": sk, "document": str(item["document"])}

    def query_partition(self, pk: str) -> list[Mapping[str, str]]:
        """Return every item in partition ``pk``."""

        from boto3.dynamodb.conditions import Key

        found: list[Mapping[str, str]] = []
        start: dict[str, str] | None = None
        while True:
            kwargs: dict[str, object] = {
                "KeyConditionExpression": Key("pk").eq(pk),
                "ConsistentRead": True,
            }
            if start is not None:
                kwargs["ExclusiveStartKey"] = start
            page = self._table.query(**kwargs)
            raw_items = page.get("Items", [])
            if isinstance(raw_items, list):
                for item in raw_items:
                    if not isinstance(item, Mapping) or "document" not in item:
                        continue
                    found.append(
                        {
                            "pk": str(item.get("pk", pk)),
                            "sk": str(item.get("sk", "")),
                            "document": str(item["document"]),
                        }
                    )
            last = page.get("LastEvaluatedKey")
            if not isinstance(last, Mapping):
                break
            start = {"pk": str(last.get("pk", pk)), "sk": str(last.get("sk", ""))}
        return found


def open_metadata_table(settings: Settings) -> BotoMetadataTable:
    """Open the metadata table with the execution role. No access keys are passed."""

    name = settings.metadata_table_name.strip()
    if not name:
        raise ValueError("METADATA_TABLE_NAME is required when EXECUTION_MODE is aws.")
    import boto3

    resource = boto3.resource("dynamodb", region_name=settings.aws_region)
    table = resource.Table(name)
    return BotoMetadataTable(cast(DynamoItemTable, table))


class DynamoRepository:
    """Read and write domain rows for one tenant in one DynamoDB metadata table."""

    def __init__(self, table: MetadataTable, tenant_id: str = LOCAL_TENANT) -> None:
        self._table = table
        self._tenant_id = validate_tenant_id(tenant_id)
        self._tracked: dict[tuple[str, str], object] = {}

    @property
    def tenant_id(self) -> str:
        """Return the tenant this repository is allowed to see."""

        return self._tenant_id

    def add(self, row: object) -> None:
        """Insert ``row`` for this tenant and make its key available."""

        stamp_tenant(row, self._tenant_id)
        self._ensure_key(row)
        self._tracked[self._key(row)] = row
        self._write(row)

    def save(self) -> None:
        """Persist pending updates."""

        for row in list(self._tracked.values()):
            self._write(row)

    def get_dataset(self, dataset_id: str) -> DatasetRow | None:
        """Return one dataset, or none when the id is unknown."""

        return self._get(DatasetRow, dataset_id)

    def list_datasets(self) -> list[DatasetRow]:
        """Return datasets in registration order."""

        return _by_created(self._rows(DatasetRow))

    def next_dataset_version(self, name: str) -> str:
        """Return the next monotonic version label for ``name``."""

        numbers = [
            int(row.version)
            for row in self._rows(DatasetRow)
            if row.name == name and row.version.isdigit()
        ]
        return str(max(numbers, default=0) + 1)

    def get_training_run(self, run_id: str) -> TrainingRunRow | None:
        """Return one training run."""

        return self._get(TrainingRunRow, run_id)

    def list_training_runs(self) -> list[TrainingRunRow]:
        """Return training runs in creation order."""

        return _by_created(self._rows(TrainingRunRow))

    def get_model(self, model_id: str) -> ModelVersionRow | None:
        """Return one model version."""

        return self._get(ModelVersionRow, model_id)

    def list_models(self) -> list[ModelVersionRow]:
        """Return model versions in registration order."""

        return _by_created(self._rows(ModelVersionRow))

    def next_model_version(self, family: str) -> str:
        """Return the next version label for ``family``."""

        numbers = [
            int(row.version)
            for row in self._rows(ModelVersionRow)
            if row.model_family == family and row.version.isdigit()
        ]
        return str(max(numbers, default=0) + 1)

    def production_model(self) -> ModelVersionRow | None:
        """Return the current production model, if one exists."""

        rows = [row for row in self._rows(ModelVersionRow) if row.status == "PRODUCTION"]
        if not rows:
            return None
        return max(rows, key=lambda row: (as_utc(row.created_at), row.id))

    def production_for_family(self, family: str) -> list[ModelVersionRow]:
        """Return production models of ``family``."""

        rows = [
            row
            for row in self._rows(ModelVersionRow)
            if row.model_family == family and row.status == "PRODUCTION"
        ]
        return _by_created(rows)

    def add_points(self, points: list[ForecastPointRow]) -> None:
        """Insert forecast points."""

        for point in points:
            self.add(point)

    def list_points(self, forecast_run_id: str) -> list[ForecastPointRow]:
        """Return points for one forecast, ordered by series and date."""

        rows = [
            row for row in self._rows(ForecastPointRow) if row.forecast_run_id == forecast_run_id
        ]
        return sorted(rows, key=lambda row: (row.series_id, row.date, row.id or 0))

    def get_forecast(self, forecast_id: str) -> ForecastRunRow | None:
        """Return one forecast run."""

        return self._get(ForecastRunRow, forecast_id)

    def get_forecast_by_idempotency_key(self, key: str) -> ForecastRunRow | None:
        """Return the forecast stored for ``key``, if the client sent one before."""

        item = self._table.get_item(_IDEMPOTENCY, self._idempotency_key(key))
        if item is not None:
            payload = _loads(item["document"])
            forecast_id = payload.get("forecast_id")
            if isinstance(forecast_id, str):
                found = self.get_forecast(forecast_id)
                if found is not None:
                    return found
        for row in self._rows(ForecastRunRow):
            if row.idempotency_key == key:
                return row
        return None

    def count_forecasts_created_on(self, day: date) -> int:
        """Return how many forecast runs were requested on ``day`` in UTC."""

        return sum(1 for row in self._rows(ForecastRunRow) if as_utc(row.created_at).date() == day)

    def list_forecasts(self) -> list[ForecastRunRow]:
        """Return forecast runs in request order."""

        return _by_created(self._rows(ForecastRunRow))

    def find_explanation(
        self,
        forecast_run_id: str,
        scope_key: str,
        prompt_version: str,
    ) -> AIExplanationRow | None:
        """Return the stored explanation for this forecast, scope, and prompt."""

        matches = [
            row
            for row in self._rows(AIExplanationRow)
            if row.forecast_run_id == forecast_run_id
            and row.scope_key == scope_key
            and row.prompt_version == prompt_version
            and row.status == "valid"
        ]
        if not matches:
            return None
        return max(matches, key=lambda row: (as_utc(row.created_at), row.id))

    def latest_explanation(self, forecast_run_id: str, scope_key: str) -> AIExplanationRow | None:
        """Return the newest valid explanation for this forecast and scope."""

        matches = [
            row
            for row in self._rows(AIExplanationRow)
            if row.forecast_run_id == forecast_run_id
            and row.scope_key == scope_key
            and row.status == "valid"
        ]
        if not matches:
            return None
        return max(matches, key=lambda row: (as_utc(row.created_at), row.id))

    def count_explanation_calls_on(self, day: date) -> int:
        """Return explanation attempts recorded on ``day`` in UTC."""

        return self.explanation_usage_on(day).calls

    def explanation_usage_on(self, day: date) -> ExplanationUsageTotals:
        """Return explanation attempts and token totals recorded on ``day``."""

        calls = 0
        input_tokens = 0
        output_tokens = 0
        for row in self._rows(AIExplanationRow):
            if as_utc(row.created_at).date() != day:
                continue
            calls += 1
            input_tokens += row.input_tokens
            output_tokens += row.output_tokens
        return ExplanationUsageTotals(
            calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def latest_forecast_error(self) -> ForecastErrorEvaluationRow | None:
        """Return the newest forecast-error score."""

        return _latest(self._rows(ForecastErrorEvaluationRow))

    def latest_monitoring_report(self) -> MonitoringReportRow | None:
        """Return the newest drift monitoring report."""

        return _latest(self._rows(MonitoringReportRow))

    def latest_succeeded_forecast(self) -> ForecastRunRow | None:
        """Return the newest forecast that finished successfully."""

        rows = [row for row in self._rows(ForecastRunRow) if row.status == "SUCCEEDED"]
        return _latest(rows)

    def get_retrain_request(self, request_id: str) -> RetrainRequestRow | None:
        """Return one retrain request."""

        return self._get(RetrainRequestRow, request_id)

    def pending_retrain_request(self) -> RetrainRequestRow | None:
        """Return the open retrain request, if a person has not confirmed one yet."""

        rows = [row for row in self._rows(RetrainRequestRow) if row.status == "PENDING"]
        if not rows:
            return None
        return max(rows, key=lambda row: (as_utc(row.requested_at), row.id))

    def list_retrain_requests(self) -> list[RetrainRequestRow]:
        """Return retrain requests in creation order."""

        return _by_created(self._rows(RetrainRequestRow))

    def list_decisions(self, candidate_id: str) -> list[PromotionDecisionRow]:
        """Return promotion decisions for one model."""

        rows = [row for row in self._rows(PromotionDecisionRow) if row.candidate_id == candidate_id]
        return sorted(rows, key=lambda row: (as_utc(row.created_at), row.id))

    def _get[RowT: Base](self, model: type[RowT], row_id: str) -> RowT | None:
        key = (model.__tablename__, row_id)
        tracked = self._tracked.get(key)
        if tracked is not None:
            row = cast(RowT, tracked)
            return row if self._visible(row) else None
        item = self._table.get_item(key[0], key[1])
        if item is None:
            return None
        row = row_from_document(model, _loads(item["document"]))
        if not self._visible(row):
            return None
        self._tracked[key] = row
        return row

    def _rows[RowT: Base](self, model: type[RowT]) -> list[RowT]:
        table = model.__tablename__
        loaded: list[RowT] = []
        seen: set[tuple[str, str]] = set()
        for item in self._table.query_partition(table):
            row = row_from_document(model, _loads(item["document"]))
            if not self._visible(row):
                continue
            key = (table, str(_primary_key(row)))
            seen.add(key)
            current = self._tracked.get(key)
            if current is None:
                self._tracked[key] = row
                loaded.append(row)
            else:
                loaded.append(cast(RowT, current))
        for tracked_key, tracked in self._tracked.items():
            if tracked_key[0] == table and tracked_key not in seen:
                loaded.append(cast(RowT, tracked))
        return loaded

    def _ensure_key(self, row: object) -> None:
        column = mapper_for(row).primary_key[0]
        key = _column_key(column)
        if getattr(row, key) is not None:
            return
        if not isinstance(column.type, Integer):
            raise ValueError("Metadata row is missing its id.")
        setattr(row, key, self._allocate_id(_table_name(row)))

    def _allocate_id(self, table: str) -> int:
        current = self._table.get_item(_META, f"{table}_seq")
        number = 1
        if current is not None:
            payload = _loads(current["document"])
            stored = payload.get("value")
            if isinstance(stored, int) and not isinstance(stored, bool):
                number = stored + 1
        self._table.put_item(_META, f"{table}_seq", json.dumps({"value": number}))
        return number

    def _write(self, row: object) -> None:
        key = self._key(row)
        self._table.put_item(key[0], key[1], json.dumps(document_from_row(row)))
        if isinstance(row, ForecastRunRow) and row.idempotency_key:
            pointer = json.dumps({"forecast_id": row.id})
            self._table.put_item(_IDEMPOTENCY, self._idempotency_key(row.idempotency_key), pointer)

    def _key(self, row: object) -> tuple[str, str]:
        return (_table_name(row), str(_primary_key(row)))

    def _visible(self, row: object) -> bool:
        return row_tenant_id(row) == self._tenant_id

    def _idempotency_key(self, key: str) -> str:
        return f"{self._tenant_id}#{key}"


def _table_name(row: object) -> str:
    table = mapper_for(row).local_table
    if not isinstance(table, Table):
        raise TypeError("Metadata row is missing its table name.")
    return table.name


def _primary_key(row: object) -> object:
    return getattr(row, _column_key(mapper_for(row).primary_key[0]))


def _column_key(column: Any) -> str:
    key = column.key
    if not isinstance(key, str):
        raise TypeError("Metadata column is missing its key.")
    return key


def _loads(document: str) -> dict[str, object]:
    payload = json.loads(document)
    if not isinstance(payload, dict):
        raise ValueError("Metadata document must be an object.")
    return {str(key): value for key, value in payload.items()}


def _by_created[RowT: Base](rows: list[RowT]) -> list[RowT]:
    return sorted(rows, key=lambda row: (as_utc(_created_at(row)), str(_primary_key(row))))


def _latest[RowT: Base](rows: list[RowT]) -> RowT | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (as_utc(_created_at(row)), str(_primary_key(row))))


def _created_at(row: Base) -> datetime:
    value = cast(_Stamped, row).created_at
    if not isinstance(value, datetime):
        raise ValueError("Metadata row is missing created_at.")
    return value


class _Stamped(Protocol):
    created_at: datetime

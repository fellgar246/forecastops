"""SQL access for domain rows.

Services and the local job runner call these methods. They do not issue SQL themselves.
"""

from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from forecastops_api.persistence import (
    DatasetRow,
    ForecastPointRow,
    ForecastRunRow,
    ModelVersionRow,
    PromotionDecisionRow,
    TrainingRunRow,
)


class Repository:
    """Read and write domain rows for one database session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, row: object) -> None:
        """Insert ``row`` and flush so its key is available."""

        self._session.add(row)
        self._session.flush()

    def save(self) -> None:
        """Flush pending updates."""

        self._session.flush()

    def get_dataset(self, dataset_id: str) -> DatasetRow | None:
        """Return one dataset, or none when the id is unknown."""

        return self._session.get(DatasetRow, dataset_id)

    def list_datasets(self) -> list[DatasetRow]:
        """Return datasets in registration order."""

        statement = select(DatasetRow).order_by(DatasetRow.created_at, DatasetRow.id)
        return list(self._session.scalars(statement))

    def next_dataset_version(self, name: str) -> str:
        """Return the next monotonic version label for ``name``."""

        statement = select(DatasetRow.version).where(DatasetRow.name == name)
        labels = list(self._session.scalars(statement))
        numbers = [int(label) for label in labels if label.isdigit()]
        return str(max(numbers, default=0) + 1)

    def get_training_run(self, run_id: str) -> TrainingRunRow | None:
        """Return one training run."""

        return self._session.get(TrainingRunRow, run_id)

    def list_training_runs(self) -> list[TrainingRunRow]:
        """Return training runs in creation order."""

        statement = select(TrainingRunRow).order_by(TrainingRunRow.created_at, TrainingRunRow.id)
        return list(self._session.scalars(statement))

    def get_model(self, model_id: str) -> ModelVersionRow | None:
        """Return one model version."""

        return self._session.get(ModelVersionRow, model_id)

    def list_models(self) -> list[ModelVersionRow]:
        """Return model versions in registration order."""

        statement = select(ModelVersionRow).order_by(ModelVersionRow.created_at, ModelVersionRow.id)
        return list(self._session.scalars(statement))

    def next_model_version(self, family: str) -> str:
        """Return the next version label for ``family``."""

        statement = select(ModelVersionRow.version).where(ModelVersionRow.model_family == family)
        labels = list(self._session.scalars(statement))
        numbers = [int(label) for label in labels if label.isdigit()]
        return str(max(numbers, default=0) + 1)

    def production_model(self) -> ModelVersionRow | None:
        """Return the current production model, if one exists."""

        statement = (
            select(ModelVersionRow)
            .where(ModelVersionRow.status == "PRODUCTION")
            .order_by(ModelVersionRow.created_at.desc())
        )
        return self._session.scalars(statement).first()

    def production_for_family(self, family: str) -> list[ModelVersionRow]:
        """Return production models of ``family``."""

        statement = select(ModelVersionRow).where(
            ModelVersionRow.model_family == family,
            ModelVersionRow.status == "PRODUCTION",
        )
        return list(self._session.scalars(statement))

    def add_points(self, points: list[ForecastPointRow]) -> None:
        """Insert forecast points."""

        self._session.add_all(points)
        self._session.flush()

    def list_points(self, forecast_run_id: str) -> list[ForecastPointRow]:
        """Return points for one forecast, ordered by series and date."""

        statement = (
            select(ForecastPointRow)
            .where(ForecastPointRow.forecast_run_id == forecast_run_id)
            .order_by(ForecastPointRow.series_id, ForecastPointRow.date)
        )
        return list(self._session.scalars(statement))

    def get_forecast(self, forecast_id: str) -> ForecastRunRow | None:
        """Return one forecast run."""

        return self._session.get(ForecastRunRow, forecast_id)

    def get_forecast_by_idempotency_key(self, key: str) -> ForecastRunRow | None:
        """Return the forecast stored for ``key``, if the client sent one before."""

        statement = select(ForecastRunRow).where(ForecastRunRow.idempotency_key == key)
        return self._session.scalars(statement).first()

    def count_forecasts_created_on(self, day: date) -> int:
        """Return how many forecast runs were requested on ``day`` in UTC."""

        total = 0
        for row in self.list_forecasts():
            if _utc(row.created_at).date() == day:
                total += 1
        return total

    def list_forecasts(self) -> list[ForecastRunRow]:
        """Return forecast runs in request order."""

        statement = select(ForecastRunRow).order_by(ForecastRunRow.created_at, ForecastRunRow.id)
        return list(self._session.scalars(statement))

    def list_decisions(self, candidate_id: str) -> list[PromotionDecisionRow]:
        """Return promotion decisions for one model."""

        statement = (
            select(PromotionDecisionRow)
            .where(PromotionDecisionRow.candidate_id == candidate_id)
            .order_by(PromotionDecisionRow.created_at)
        )
        return list(self._session.scalars(statement))


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)

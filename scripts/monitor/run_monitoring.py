"""Run drift monitoring on this machine.

The command stores a report and, when retraining is recommended, a confirmation
request. It does not start training and it does not approve a model.
"""

import argparse
import sys
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from forecastops_api.artifacts import select_artifact_store
from forecastops_api.errors import ApiError
from forecastops_api.forecast_jobs import select_batch_inference
from forecastops_api.monitoring import MonitoringService
from forecastops_api.registry import select_model_registry
from forecastops_api.repositories import Repository
from forecastops_api.schedules import ScheduleService
from forecastops_api.services import ForecastService
from forecastops_api.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    """Compare the recent window with the training baseline."""

    parser = argparse.ArgumentParser(description="Run drift monitoring locally.")
    parser.add_argument("--as-of", dest="as_of", default=None)
    args = parser.parse_args(argv)
    try:
        as_of = None if args.as_of is None else date.fromisoformat(args.as_of)
    except ValueError:
        print("as-of must be a calendar date, such as 2026-02-25.", file=sys.stderr)
        return 1
    settings = get_settings()
    engine = create_engine(settings.database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        result = _service(session).run(as_of)
        session.commit()
    except ApiError as exc:
        session.rollback()
        print(exc.body.model_dump_json(), file=sys.stderr)
        return 1
    finally:
        session.close()
        engine.dispose()
    print(result.model_dump_json())
    return 0


def _service(session: Session) -> MonitoringService:
    settings = get_settings()
    repository = Repository(session)
    forecasts = ForecastService(
        repository,
        settings,
        select_artifact_store(settings),
        select_model_registry(settings),
        batch=select_batch_inference(settings),
    )
    return MonitoringService(repository, ScheduleService(forecasts, repository), settings)


if __name__ == "__main__":
    raise SystemExit(main())

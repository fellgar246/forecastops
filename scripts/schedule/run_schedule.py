"""Run one scheduled operation on this machine.

The command calls the forecast service directly. It does not construct a
cloud scheduler client.
"""

import argparse
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from forecastops_api.artifacts import select_artifact_store
from forecastops_api.errors import ApiError
from forecastops_api.forecast_jobs import select_batch_inference
from forecastops_api.registry import select_model_registry
from forecastops_api.repositories import Repository
from forecastops_api.schedules import SCHEDULE_ACTIONS, ScheduleService
from forecastops_api.services import ForecastService
from forecastops_api.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    """Run ``daily_forecast``, ``weekly_evaluation``, or ``monthly_retrain``."""

    parser = argparse.ArgumentParser(description="Run one scheduled operation locally.")
    parser.add_argument("action", choices=SCHEDULE_ACTIONS)
    args = parser.parse_args(argv)
    settings = get_settings()
    engine = create_engine(settings.database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        result = _service(session).execute(args.action)
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


def _service(session: Session) -> ScheduleService:
    settings = get_settings()
    forecasts = ForecastService(
        Repository(session),
        settings,
        select_artifact_store(settings),
        select_model_registry(settings),
        batch=select_batch_inference(settings),
    )
    return ScheduleService(forecasts, Repository(session))


if __name__ == "__main__":
    raise SystemExit(main())

#!/bin/sh
set -eu

alembic -c apps/api/alembic.ini upgrade head
exec uvicorn forecastops_api.main:app --host 0.0.0.0 --port 8000

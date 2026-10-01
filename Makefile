.PHONY: test lint seed-data profile-data train-deepar start-pipeline refresh-forecast evaluate-forecasts evaluate-suite request-retrain monitor-drift aws-plan aws-deploy aws-status aws-cost-check aws-destroy aws-clean-artifacts aws-smoke

test:
	uv run pytest
	npm --prefix apps/web test

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	npm --prefix apps/web run lint
	npm --prefix apps/web run typecheck
	terraform fmt -check -recursive infra

OUT ?= data/synthetic
PROFILE ?= default
DATASET ?= data/synthetic
DATASET_VERSION ?=
GIT_SHA ?=
CONFIG ?=

seed-data:
	uv run python scripts/seed/seed_data.py --output $(OUT) --profile $(PROFILE)

profile-data:
	uv run python scripts/profile/profile_data.py --dataset $(OUT)

train-deepar:
	uv run python scripts/train/train_deepar.py --dataset $(DATASET)

start-pipeline:
	uv run python scripts/train/start_pipeline.py --dataset-version "$(DATASET_VERSION)" --git-sha "$(GIT_SHA)" --config "$(CONFIG)"

refresh-forecast:
	uv run python scripts/schedule/run_schedule.py daily_forecast

evaluate-forecasts:
	uv run python scripts/schedule/run_schedule.py weekly_evaluation

evaluate-suite:
	uv run python scripts/evaluate/run_suite.py --output var/evaluation-report.json

request-retrain:
	uv run python scripts/schedule/run_schedule.py monthly_retrain

monitor-drift:
	uv run python scripts/monitor/run_monitoring.py

aws-plan:
	./scripts/aws/plan.sh

aws-deploy:
	./scripts/aws/deploy.sh

aws-status:
	./scripts/aws/status.sh

aws-cost-check:
	./scripts/aws/cost-check.sh

aws-destroy:
	./scripts/aws/destroy.sh

aws-clean-artifacts:
	./scripts/aws/clean-artifacts.sh

aws-smoke:
	./scripts/aws/smoke.sh

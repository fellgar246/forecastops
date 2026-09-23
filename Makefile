.PHONY: test lint seed-data profile-data aws-plan aws-deploy aws-status aws-cost-check aws-destroy aws-clean-artifacts

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

seed-data:
	uv run python scripts/seed/seed_data.py --output $(OUT) --profile $(PROFILE)

profile-data:
	uv run python scripts/profile/profile_data.py --dataset $(OUT)

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

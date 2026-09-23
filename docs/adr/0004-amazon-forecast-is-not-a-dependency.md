# ADR 0004: Amazon Forecast is not a dependency

## Context

A managed forecast product would own training, evaluation, and artifacts. This platform is meant to own that path itself, and another forecast service would be a second system to keep compatible.

## Decision

The platform does not call Amazon Forecast. Forecasts are produced by the models in this repository.

## Consequences

Feature code, backtests, and artifacts stay in this codebase. There is no client, permission, or data mapping for that service.

# ADR 0007: Artifact storage is one interface

## Context

Forecasting code needs to store datasets, models, evaluations, and forecasts. Local development uses the filesystem. A cloud profile uses object storage. Those paths should not fork the forecasting logic.

## Decision

Domain services read and write bytes through one artifact store. The settings object selects a directory on disk when AWS is disabled, and one remote bucket when AWS is enabled. Keys use only the prefixes `raw/`, `processed/`, `features/`, `training/`, `models/`, `forecasts/`, and `evaluations/`. The remote bucket blocks public access, encrypts objects, and expires non-current versions. The application does not log object bodies. Local dataset upload is a direct multipart request. When AWS is enabled, the API issues a pre-signed upload URL.

## Consequences

Contract tests cover both implementations. Local mode does not construct a remote client. Replacing the disk with object storage does not change training or forecast logic. The infrastructure module describes the bucket rules and is not applied by the local workflow.

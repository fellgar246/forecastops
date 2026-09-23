# ADR 0005: Model approval requires a person

## Context

A training job can finish without error and still be worse than the model already in use. Automatic promotion would put that candidate into production before anyone reviewed the evidence.

## Decision

A finished training job is not an approved model. Promotion requires the quality gate and a person's decision. Automatic retraining may recommend a candidate. It must not deploy one. The gate itself may reject a candidate when thresholds fail. Entering approved or rejected from the pending state is a human action.

## Consequences

Production changes only after someone reviews the evidence. The system can refuse a weak candidate on its own. It cannot accept one on its own.

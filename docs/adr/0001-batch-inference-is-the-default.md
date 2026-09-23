# ADR 0001: Batch inference is the default

## Context

An always-on inference endpoint keeps billing while nobody is asking for a forecast. Retail replenishment can use demand that was produced ahead of time and stored.

## Decision

Forecasts are produced by batch jobs and stored as artifacts. The API serves those artifacts. Online inference stays off. A serverless endpoint is optional and stays disabled unless a demonstration turns it on deliberately.

## Consequences

Users see the latest completed batch rather than a live model call. Request latency is the age of that batch. The platform does not pay for an always-on endpoint in the default profile.

# ADR 0003: The language model explains model output

## Context

A language model can write fluent text that looks like a forecast while inventing numbers, promotions, or causes. Those sentences would be indistinguishable from model output if they were allowed to change the forecast.

## Decision

Numerical demand forecasts come only from forecasting models. A language model may receive a structured forecast package and write an explanation of that package. It must not invent or change forecast numbers, promotions, causal claims, or certainty. Generated text is labeled as an explanation and talks about signals, not causes.

## Consequences

Forecast quality is measured on the forecasting model. An explanation failure cannot change the numbers the API serves. Calling the language model is optional and is covered by its own kill switch.

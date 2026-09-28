import { expect, test } from "vitest";
import { ZodError } from "zod";
import { parseHealthResponse } from "./health";

test("parses a local health payload", () => {
  const health = parseHealthResponse({
    status: "healthy",
    execution_mode: "local",
    aws_enabled: false,
    bedrock_enabled: false,
    sagemaker_enabled: false,
    training_enabled: true,
    max_forecast_horizon_days: 90,
    max_training_jobs_per_day: 2,
  });

  expect(health.execution_mode).toBe("local");
  expect(health.aws_enabled).toBe(false);
  expect(health.max_forecast_horizon_days).toBe(90);
});

test("rejects an unknown execution mode", () => {
  expect(() =>
    parseHealthResponse({
      status: "healthy",
      execution_mode: "cluster",
      aws_enabled: false,
      bedrock_enabled: false,
      sagemaker_enabled: false,
      training_enabled: true,
      max_forecast_horizon_days: 90,
      max_training_jobs_per_day: 2,
    }),
  ).toThrow(ZodError);
});

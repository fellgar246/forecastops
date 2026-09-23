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
  });

  expect(health.execution_mode).toBe("local");
  expect(health.aws_enabled).toBe(false);
});

test("rejects an unknown execution mode", () => {
  expect(() =>
    parseHealthResponse({
      status: "healthy",
      execution_mode: "cluster",
      aws_enabled: false,
      bedrock_enabled: false,
      sagemaker_enabled: false,
    }),
  ).toThrow(ZodError);
});

import { z } from "zod";

export const healthResponseSchema = z.object({
  status: z.literal("healthy"),
  execution_mode: z.enum(["local", "aws"]),
  aws_enabled: z.boolean(),
  bedrock_enabled: z.boolean(),
  sagemaker_enabled: z.boolean(),
  training_enabled: z.boolean(),
  max_forecast_horizon_days: z.number().int().positive(),
  max_training_jobs_per_day: z.number().int().nonnegative(),
});

export type HealthResponse = z.infer<typeof healthResponseSchema>;

export function parseHealthResponse(payload: unknown): HealthResponse {
  return healthResponseSchema.parse(payload);
}

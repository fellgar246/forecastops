import { z } from "zod";

export const healthResponseSchema = z.object({
  status: z.literal("healthy"),
  execution_mode: z.enum(["local", "aws"]),
  aws_enabled: z.boolean(),
  bedrock_enabled: z.boolean(),
  sagemaker_enabled: z.boolean(),
});

export type HealthResponse = z.infer<typeof healthResponseSchema>;

export function parseHealthResponse(payload: unknown): HealthResponse {
  return healthResponseSchema.parse(payload);
}

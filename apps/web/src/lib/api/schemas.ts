import { z } from "zod";

export const modelFamilySchema = z.enum([
  "naive",
  "seasonal_naive",
  "holt_winters",
  "gradient_boosting",
  "deepar",
]);

export const datasetStatusSchema = z.enum(["registered", "valid", "invalid"]);
export const trainingStatusSchema = z.enum([
  "QUEUED",
  "PREPROCESSING",
  "TRAINING",
  "EVALUATING",
  "REGISTERING",
  "COMPLETED",
  "FAILED",
]);
export const modelStatusSchema = z.enum([
  "TRAINED",
  "EVALUATED",
  "PENDING_APPROVAL",
  "APPROVED",
  "PRODUCTION",
  "REJECTED",
]);
export const forecastStatusSchema = z.enum(["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]);
export const granularitySchema = z.enum(["day", "week"]);

export const errorBodySchema = z.object({
  code: z.string(),
  message: z.string(),
  details: z.record(z.string(), z.unknown()).default({}),
});

export const acceptedJobSchema = z.object({
  job_id: z.string(),
});

const metricSummarySchema = z.object({
  row_count: z.number(),
  mae: z.number(),
  rmse: z.number(),
  wape: z.number(),
  smape: z.number(),
  bias: z.number(),
  pinball_loss_p10: z.number().nullable(),
  pinball_loss_p90: z.number().nullable(),
});

export const modelMetricsSchema = metricSummarySchema
  .extend({
    p90_coverage: z.number().nullable().optional(),
  })
  .nullable();

const sliceSchema = z.object({
  key: z.string(),
  metrics: metricSummarySchema,
});

export const evaluationReportSchema = z
  .object({
    dataset_version: z.string().nullable(),
    fold_count: z.number(),
    horizon: z.number(),
    model_family: z.string(),
    runtime_ms: z.number(),
    metrics: z.object({
      global: metricSummarySchema,
      by_category: z.array(sliceSchema),
      by_store: z.array(sliceSchema),
      by_horizon_step: z.array(sliceSchema),
      by_demand_quartile: z.array(sliceSchema),
    }),
    wape_by_horizon: z.array(z.object({ step: z.number(), wape: z.number() })),
    skipped_series: z.array(z.object({ series_id: z.string(), reason: z.string() })),
  })
  .nullable();

const findingSchema = z.object({
  code: z.string(),
  count: z.number(),
  message: z.string(),
  rate: z.number().optional(),
});

export const qualityReportSchema = z
  .object({
    status: datasetStatusSchema,
    blocking: z.array(findingSchema),
    advisory: z.array(findingSchema),
    rates: z.object({
      duplicates: z.number(),
      missing_values: z.number(),
      stockouts: z.number(),
    }),
  })
  .nullable();

export const datasetSchema = z.object({
  id: z.string(),
  name: z.string(),
  version: z.string(),
  source: z.enum(["synthetic", "upload"]),
  status: datasetStatusSchema,
  uri: z.string(),
  schema_version: z.string(),
  row_count: z.number(),
  date_min: z.string().nullable(),
  date_max: z.string().nullable(),
  quality_report: qualityReportSchema,
  created_at: z.string(),
});

export const datasetListSchema = z.object({
  items: z.array(datasetSchema),
});

export const catalogSchema = z.object({
  stores: z.array(z.object({ id: z.string(), name: z.string() })),
  categories: z.array(z.object({ id: z.string(), name: z.string() })),
  skus: z.array(z.object({ id: z.string(), category_id: z.string() })),
});

export const trainingRunSchema = z.object({
  id: z.string(),
  dataset_id: z.string(),
  dataset_version: z.string(),
  model_family: z.string(),
  configuration: z.record(z.string(), z.unknown()),
  status: trainingStatusSchema,
  started_at: z.string().nullable(),
  finished_at: z.string().nullable(),
  artifact_uri: z.string(),
  metrics: evaluationReportSchema,
  error_message: z.string().nullable(),
  git_sha: z.string(),
  pipeline_execution_arn: z.string(),
  created_at: z.string(),
});

export const trainingListSchema = z.object({
  items: z.array(trainingRunSchema),
});

export const promotionSchema = z
  .object({
    reference_id: z.string(),
    thresholds: z.record(z.string(), z.number()),
    checks: z.object({
      wape_improved: z.boolean(),
      bias_within_limit: z.boolean(),
      coverage_within_limit: z.boolean().nullable(),
      no_critical_segment_regression: z.boolean(),
    }),
    reason: z.string().nullable(),
    p90_coverage: z.number().nullable(),
    regressed_categories: z.array(
      z.object({
        category_id: z.string(),
        candidate_wape: z.number(),
        reference_wape: z.number(),
      }),
    ),
  })
  .nullable();

export const modelSchema = z.object({
  id: z.string(),
  model_family: z.string(),
  version: z.string(),
  training_run_id: z.string(),
  dataset_id: z.string(),
  dataset_version: z.string(),
  registry_arn: z.string(),
  registry_status: z.string(),
  status: modelStatusSchema,
  metrics: modelMetricsSchema,
  rejection_reason: z.string().nullable(),
  promotion: promotionSchema,
  approved_at: z.string().nullable(),
  created_at: z.string(),
});

export const modelListSchema = z.object({
  items: z.array(modelSchema),
});

export const forecastSchema = z.object({
  id: z.string(),
  model_id: z.string(),
  model_family: z.string(),
  model_version: z.string(),
  dataset_id: z.string(),
  dataset_version: z.string(),
  horizon: z.number(),
  granularity: granularitySchema,
  status: forecastStatusSchema,
  output_uri: z.string(),
  error_message: z.string().nullable(),
  created_at: z.string(),
});

export const forecastListSchema = z.object({
  items: z.array(forecastSchema),
});

const forecastPointSchema = z.object({
  series_id: z.string(),
  date: z.string(),
  p10: z.number().nullable(),
  p50: z.number(),
  p90: z.number().nullable(),
  actual: z.number().nullable(),
});

export const forecastSeriesSchema = z.object({
  items: z.array(forecastPointSchema),
  series: z.array(
    z.object({
      series_id: z.string(),
      point_count: z.number(),
      p10_total: z.number().nullable(),
      p50_total: z.number().nullable(),
      p90_total: z.number().nullable(),
    }),
  ),
  history: z.array(z.object({ date: z.string(), actual: z.number() })),
  daily: z.array(
    z.object({
      date: z.string(),
      p10: z.number().nullable(),
      p50: z.number().nullable(),
      p90: z.number().nullable(),
      actual: z.number().nullable(),
    }),
  ),
  cutoff: z.string().nullable(),
  p10_total: z.number().nullable(),
  p50_total: z.number().nullable(),
  p90_total: z.number().nullable(),
});

export const dataQualitySchema = z.object({
  items: z.array(
    z.object({
      dataset_id: z.string(),
      dataset_version: z.string(),
      status: datasetStatusSchema,
      quality_report: qualityReportSchema,
    }),
  ),
});

export const awsHealthSchema = z.object({
  aws_enabled: z.boolean(),
  aws_ml_enabled: z.boolean(),
  bedrock_enabled: z.boolean(),
  sagemaker_enabled: z.boolean(),
  online_inference: z.boolean(),
});

export const explanationSchema = z.object({
  id: z.string(),
  model_id: z.string(),
  prompt_version: z.string(),
  summary: z.string(),
  signals: z.array(
    z.object({
      name: z.string(),
      direction: z.enum(["positive", "negative"]).optional(),
      value: z.number().nullable().optional(),
      kind: z.enum(["driver", "context"]).optional(),
    }),
  ),
  risks: z.array(z.string()),
  uncertainty: z.string(),
  checks: z.array(z.string()),
  generated_at: z.string(),
  package: z.record(z.string(), z.unknown()).optional(),
});

export type Dataset = z.infer<typeof datasetSchema>;
export type DatasetCatalog = z.infer<typeof catalogSchema>;
export type TrainingRun = z.infer<typeof trainingRunSchema>;
export type ModelVersion = z.infer<typeof modelSchema>;
export type ForecastRun = z.infer<typeof forecastSchema>;
export type ForecastSeries = z.infer<typeof forecastSeriesSchema>;
export type QualityReport = z.infer<typeof qualityReportSchema>;
export type DataQualityList = z.infer<typeof dataQualitySchema>;
export type Explanation = z.infer<typeof explanationSchema>;
export type ModelFamily = z.infer<typeof modelFamilySchema>;

export type SeriesQuery = {
  category?: string;
  sku?: string;
  store?: string;
  horizon?: number;
};

export type ExplanationView =
  | { kind: "disabled"; message: string }
  | { kind: "missing" }
  | { kind: "pending" }
  | { kind: "ready"; explanation: Explanation }
  | { kind: "invalid"; message: string; check: string }
  | { kind: "limited"; message: string };

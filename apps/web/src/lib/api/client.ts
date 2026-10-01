import type { ZodType } from "zod";
import { ZodError } from "zod";
import { authEnabled, readAccessToken } from "@/lib/auth";
import { healthResponseSchema, type HealthResponse } from "@/lib/health";
import {
  acceptedJobSchema,
  awsHealthSchema,
  catalogSchema,
  dataQualitySchema,
  datasetListSchema,
  datasetSchema,
  errorBodySchema,
  explanationSchema,
  forecastListSchema,
  forecastSchema,
  forecastSeriesSchema,
  modelListSchema,
  modelSchema,
  trainingListSchema,
  trainingRunSchema,
  type Dataset,
  type DatasetCatalog,
  type DataQualityList,
  type ExplanationView,
  type ForecastRun,
  type ForecastSeries,
  type ModelFamily,
  type ModelVersion,
  type SeriesQuery,
  type TrainingRun,
} from "@/lib/api/schemas";

export const LOCAL_ACTOR_ID = "local-reviewer";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;

  constructor(status: number, code: string, message: string, details: Record<string, unknown>) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export class ResponseParseError extends Error {
  constructor(cause: ZodError) {
    super("The server returned data this page does not understand.");
    this.name = "ResponseParseError";
    console.error(cause);
  }
}

export function apiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
}

export function seriesQueryString(query: SeriesQuery): string {
  const params = new URLSearchParams();
  if (query.category) {
    params.set("category", query.category);
  }
  if (query.sku) {
    params.set("sku", query.sku);
  }
  if (query.store) {
    params.set("store", query.store);
  }
  if (query.horizon != null) {
    params.set("horizon", String(query.horizon));
  }
  const text = params.toString();
  return text ? `?${text}` : "";
}

async function parse<T>(response: Response, schema: ZodType<T>): Promise<T> {
  const payload: unknown = await response.json();
  if (!response.ok) {
    const error = errorBodySchema.safeParse(payload);
    if (error.success) {
      throw new ApiError(response.status, error.data.code, error.data.message, error.data.details);
    }
    throw new ApiError(response.status, "http_error", "The request failed.", {});
  }
  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    throw new ResponseParseError(parsed.error);
  }
  return parsed.data;
}

function authorizationHeader(): Record<string, string> {
  if (!authEnabled()) {
    return {};
  }
  const token = readAccessToken();
  if (!token) {
    return {};
  }
  return { Authorization: `Bearer ${token}` };
}

async function send(path: string, init?: RequestInit): Promise<Response> {
  return fetch(`${apiBaseUrl()}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...authorizationHeader(),
      ...init?.headers,
    },
  });
}

export function getHealth(): Promise<HealthResponse> {
  return send("/health").then((response) => parse(response, healthResponseSchema));
}

export function getAwsHealth() {
  return send("/health/aws").then((response) => parse(response, awsHealthSchema));
}

export function listDatasets() {
  return send("/datasets").then((response) => parse(response, datasetListSchema));
}

export function getDataset(id: string): Promise<Dataset> {
  return send(`/datasets/${id}`).then((response) => parse(response, datasetSchema));
}

export function getCatalog(id: string): Promise<DatasetCatalog> {
  return send(`/datasets/${id}/catalog`).then((response) => parse(response, catalogSchema));
}

export function registerDataset(body: { name: string; source: "synthetic" | "upload"; uri: string }) {
  return send("/datasets", { method: "POST", body: JSON.stringify(body) }).then((response) =>
    parse(response, acceptedJobSchema),
  );
}

export function validateDataset(id: string) {
  return send(`/datasets/${id}/validate`, { method: "POST", body: JSON.stringify({}) }).then(
    (response) => parse(response, datasetSchema),
  );
}

export function listTrainingRuns() {
  return send("/training-runs").then((response) => parse(response, trainingListSchema));
}

export function getTrainingRun(id: string): Promise<TrainingRun> {
  return send(`/training-runs/${id}`).then((response) => parse(response, trainingRunSchema));
}

export function startTraining(body: {
  dataset_id: string;
  model_family: ModelFamily;
  configuration: Record<string, unknown>;
}) {
  return send("/training-runs", { method: "POST", body: JSON.stringify(body) }).then((response) =>
    parse(response, acceptedJobSchema),
  );
}

export function listModels() {
  return send("/models").then((response) => parse(response, modelListSchema));
}

export function getModel(id: string): Promise<ModelVersion> {
  return send(`/models/${id}`).then((response) => parse(response, modelSchema));
}

export function approveModel(id: string, body: { actor_id: string; promote: boolean }) {
  return send(`/models/${id}/approve`, { method: "POST", body: JSON.stringify(body) }).then(
    (response) => parse(response, modelSchema),
  );
}

export function rejectModel(id: string, body: { actor_id: string; reason: string }) {
  return send(`/models/${id}/reject`, { method: "POST", body: JSON.stringify(body) }).then(
    (response) => parse(response, modelSchema),
  );
}

export function listForecasts() {
  return send("/forecasts").then((response) => parse(response, forecastListSchema));
}

export function getForecast(id: string): Promise<ForecastRun> {
  return send(`/forecasts/${id}`).then((response) => parse(response, forecastSchema));
}

export function getForecastSeries(id: string, query: SeriesQuery): Promise<ForecastSeries> {
  return send(`/forecasts/${id}/series${seriesQueryString(query)}`).then((response) =>
    parse(response, forecastSeriesSchema),
  );
}

export function startForecast(body: {
  model_id: string;
  horizon: number;
  granularity: "day" | "week";
  dataset_id?: string;
}) {
  return send("/forecasts", { method: "POST", body: JSON.stringify(body) }).then((response) =>
    parse(response, acceptedJobSchema),
  );
}

export function modelPerformance() {
  return send("/metrics/model-performance").then((response) => parse(response, modelListSchema));
}

export function dataQuality(): Promise<DataQualityList> {
  return send("/metrics/data-quality").then((response) => parse(response, dataQualitySchema));
}

export async function getExplanation(forecastId: string): Promise<ExplanationView> {
  const response = await send(`/forecasts/${forecastId}/explanation`);
  return readExplanation(response);
}

export async function requestExplanation(forecastId: string): Promise<ExplanationView> {
  const response = await send(`/forecasts/${forecastId}/explanation`, { method: "POST" });
  return readExplanation(response);
}

async function readExplanation(response: Response): Promise<ExplanationView> {
  if (response.status === 404) {
    return { kind: "missing" };
  }
  if (response.status === 409 || response.status === 422 || response.status === 429) {
    const body = await readError(response);
    if (response.status === 409) {
      return { kind: "disabled", message: body.message };
    }
    if (response.status === 422) {
      const check = typeof body.details.check === "string" ? body.details.check : "";
      return { kind: "invalid", message: body.message, check };
    }
    return { kind: "limited", message: body.message };
  }
  const explanation = await parse(response, explanationSchema);
  return { kind: "ready", explanation };
}

async function readError(response: Response) {
  const payload: unknown = await response.json();
  const parsed = errorBodySchema.safeParse(payload);
  if (!parsed.success) {
    throw new ResponseParseError(parsed.error);
  }
  return parsed.data;
}

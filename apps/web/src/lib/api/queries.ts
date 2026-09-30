"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  approveModel,
  dataQuality,
  getAwsHealth,
  getCatalog,
  getExplanation,
  getForecast,
  getForecastSeries,
  getHealth,
  getModel,
  listDatasets,
  listForecasts,
  listModels,
  listTrainingRuns,
  modelPerformance,
  registerDataset,
  rejectModel,
  requestExplanation,
  startForecast,
  startTraining,
  validateDataset,
} from "@/lib/api/client";
import type { ModelFamily, SeriesQuery } from "@/lib/api/schemas";

const ACTIVE_TRAINING = new Set(["QUEUED", "PREPROCESSING", "TRAINING", "EVALUATING", "REGISTERING"]);
const ACTIVE_FORECAST = new Set(["QUEUED", "RUNNING"]);

export function useHealth() {
  return useQuery({ queryKey: ["health"], queryFn: getHealth });
}

export function useAwsHealth() {
  return useQuery({ queryKey: ["health-aws"], queryFn: getAwsHealth });
}

export function useDatasets() {
  return useQuery({ queryKey: ["datasets"], queryFn: listDatasets });
}

export function useCatalog(datasetId: string | undefined) {
  return useQuery({
    queryKey: ["catalog", datasetId],
    queryFn: () => getCatalog(datasetId ?? ""),
    enabled: Boolean(datasetId),
  });
}

export function useTrainingRuns() {
  return useQuery({
    queryKey: ["training-runs"],
    queryFn: listTrainingRuns,
    refetchInterval: (query) => {
      const items = query.state.data?.items ?? [];
      return items.some((run) => ACTIVE_TRAINING.has(run.status)) ? 5000 : false;
    },
  });
}

export function useModels() {
  return useQuery({ queryKey: ["models"], queryFn: listModels });
}

export function useModel(id: string | undefined) {
  return useQuery({
    queryKey: ["models", id],
    queryFn: () => getModel(id ?? ""),
    enabled: Boolean(id),
  });
}

export function useForecasts() {
  return useQuery({
    queryKey: ["forecasts"],
    queryFn: listForecasts,
    refetchInterval: (query) => {
      const items = query.state.data?.items ?? [];
      return items.some((run) => ACTIVE_FORECAST.has(run.status)) ? 5000 : false;
    },
  });
}

export function useForecast(id: string | undefined) {
  return useQuery({
    queryKey: ["forecasts", id],
    queryFn: () => getForecast(id ?? ""),
    enabled: Boolean(id),
    refetchInterval: (query) =>
      query.state.data && ACTIVE_FORECAST.has(query.state.data.status) ? 5000 : false,
  });
}

export function useForecastSeries(id: string | undefined, query: SeriesQuery) {
  return useQuery({
    queryKey: ["forecast-series", id, query],
    queryFn: () => getForecastSeries(id ?? "", query),
    enabled: Boolean(id),
  });
}

export function useModelPerformance() {
  return useQuery({ queryKey: ["model-performance"], queryFn: modelPerformance });
}

export function useDataQuality() {
  return useQuery({ queryKey: ["data-quality"], queryFn: dataQuality });
}

export function useExplanation(forecastId: string | undefined) {
  return useQuery({
    queryKey: ["explanation", forecastId],
    queryFn: () => getExplanation(forecastId ?? ""),
    enabled: Boolean(forecastId),
  });
}

export function useRequestExplanation(forecastId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => requestExplanation(forecastId),
    onSuccess: (view) => {
      client.setQueryData(["explanation", forecastId], view);
    },
  });
}

export function useRegisterDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: registerDataset,
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["datasets"] });
      await client.invalidateQueries({ queryKey: ["data-quality"] });
    },
  });
}

export function useValidateDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: validateDataset,
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["datasets"] });
      await client.invalidateQueries({ queryKey: ["data-quality"] });
    },
  });
}

export function useStartTraining() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: { dataset_id: string; model_family: ModelFamily }) =>
      startTraining({
        ...body,
        configuration: { folds: 3, horizon: 7 },
      }),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["training-runs"] });
      await client.invalidateQueries({ queryKey: ["models"] });
      await client.invalidateQueries({ queryKey: ["model-performance"] });
    },
  });
}

export function useApproveModel() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: { id: string; promote: boolean }) =>
      approveModel(input.id, { actor_id: "local-reviewer", promote: input.promote }),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["models"] });
      await client.invalidateQueries({ queryKey: ["model-performance"] });
    },
  });
}

export function useRejectModel() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => rejectModel(id, { actor_id: "local-reviewer", reason: "human" }),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["models"] });
      await client.invalidateQueries({ queryKey: ["model-performance"] });
    },
  });
}

export function useStartForecast() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: startForecast,
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["forecasts"] });
    },
  });
}

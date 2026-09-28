"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Dialog } from "@/components/dialog";
import { useToast } from "@/components/toast";
import { ApiError } from "@/lib/api/client";
import { useHealth, useModels, useStartForecast } from "@/lib/api/queries";
import { horizonChoices } from "@/lib/horizons";
import { familyLabel } from "@/lib/status";

export function GenerateForecastButton() {
  const health = useHealth();
  const models = useModels();
  const start = useStartForecast();
  const toast = useToast();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const eligible = (models.data?.items ?? []).filter(
    (model) => model.status === "APPROVED" || model.status === "PRODUCTION",
  );
  const maxDays = health.data?.max_forecast_horizon_days;
  const choices = maxDays == null ? [] : horizonChoices(maxDays);
  const [modelId, setModelId] = useState("");
  const [choiceIndex, setChoiceIndex] = useState(0);
  const selected = choices[choiceIndex] ?? choices[0];
  const disabled = eligible.length === 0 || choices.length === 0;
  const hint =
    eligible.length === 0
      ? "Approve a model before generating a forecast."
      : choices.length === 0
        ? "The horizon limit has not loaded yet."
        : undefined;
  async function submit() {
    const model = modelId || eligible[0]?.id;
    if (!model || !selected) {
      return;
    }
    try {
      const job = await start.mutateAsync({
        model_id: model,
        horizon: selected.horizon,
        granularity: selected.granularity,
      });
      toast("Forecast started");
      setOpen(false);
      router.push(`/forecasts/${job.job_id}`);
    } catch (error) {
      toast(error instanceof ApiError ? error.message : "The forecast could not be started.");
    }
  }
  return (
    <>
      <button
        type="button"
        className="btn btn-primary"
        disabled={disabled}
        aria-describedby={hint ? "generate-forecast-hint" : undefined}
        onClick={() => setOpen(true)}
      >
        Generate forecast
      </button>
      {hint ? (
        <p id="generate-forecast-hint" className="caption">
          {hint}
        </p>
      ) : null}
      <Dialog title="Generate forecast" open={open} onClose={() => setOpen(false)}>
        <label className="field">
          Model
          <select value={modelId || eligible[0]?.id || ""} onChange={(event) => setModelId(event.target.value)}>
            {eligible.map((model) => (
              <option key={model.id} value={model.id}>
                {familyLabel(model.model_family)} {model.version}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Horizon
          <select value={choiceIndex} onChange={(event) => setChoiceIndex(Number(event.target.value))}>
            {choices.map((choice, index) => (
              <option key={choice.label} value={index}>
                {choice.label}
              </option>
            ))}
          </select>
        </label>
        <div className="dialog-actions">
          <button type="button" className="btn btn-secondary" data-cancel onClick={() => setOpen(false)}>
            Cancel
          </button>
          <button type="button" className="btn btn-primary" disabled={start.isPending} onClick={() => void submit()}>
            {start.isPending ? "Starting…" : "Generate forecast"}
          </button>
        </div>
      </Dialog>
    </>
  );
}

"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { Dialog } from "@/components/dialog";
import { EmptyState } from "@/components/empty-state";
import { ErrorNotice } from "@/components/error-notice";
import { JobStepper } from "@/components/job-stepper";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { useToast } from "@/components/toast";
import { ApiError } from "@/lib/api/client";
import { useDatasets, useHealth, useModels, useStartTraining, useTrainingRuns } from "@/lib/api/queries";
import { modelFamilySchema, type ModelFamily } from "@/lib/api/schemas";
import { formatDate, formatDuration, formatId, formatRatioPercent, MISSING_MARK } from "@/lib/format";
import { familyLabel } from "@/lib/status";

const FAMILIES = modelFamilySchema.options;

export function TrainingPage() {
  const runs = useTrainingRuns();
  const datasets = useDatasets();
  const health = useHealth();
  const models = useModels();
  const start = useStartTraining();
  const toast = useToast();
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [datasetId, setDatasetId] = useState("");
  const [family, setFamily] = useState<ModelFamily>("seasonal_naive");
  const valid = (datasets.data?.items ?? []).filter((item) => item.status === "valid");
  const selectedId = params.get("run");
  const selected = runs.data?.items.find((run) => run.id === selectedId);
  const today = new Date().toISOString().slice(0, 10);
  const usedToday = (runs.data?.items ?? []).filter((run) => run.created_at.slice(0, 10) === today).length;
  const limit = health.data?.max_training_jobs_per_day;
  const remaining = limit == null ? null : Math.max(limit - usedToday, 0);
  const trainingOff = health.data?.training_enabled === false;
  async function submit() {
    const id = datasetId || valid[0]?.id;
    if (!id) {
      return;
    }
    try {
      const job = await start.mutateAsync({ dataset_id: id, model_family: family });
      toast("Training started");
      setOpen(false);
      const query = new URLSearchParams(params.toString());
      query.set("run", job.job_id);
      router.replace(`${pathname}?${query.toString()}`);
    } catch (error) {
      toast(error instanceof ApiError ? error.message : "Training could not be started.");
    }
  }
  if (runs.isPending) {
    return (
      <main className="page">
        <PageHeader title="Training" description="Local training runs and where each one stopped." />
        <p className="skeleton-block" aria-hidden="true" />
      </main>
    );
  }
  if (runs.error) {
    return (
      <main className="page">
        <PageHeader title="Training" description="Local training runs and where each one stopped." />
        <ErrorNotice error={runs.error} onRetry={() => void runs.refetch()} />
      </main>
    );
  }
  return (
    <main className="page">
      <PageHeader
        title="Training"
        description="Local training runs and where each one stopped."
        actions={
          <button
            type="button"
            className="btn btn-primary"
            disabled={trainingOff || remaining === 0 || valid.length === 0}
            title={trainingOff ? "Training is turned off in this environment" : undefined}
            onClick={() => setOpen(true)}
          >
            Start training
          </button>
        }
      />
      {trainingOff ? <p className="notice notice-warning">Training is turned off in this environment.</p> : null}
      {(runs.data?.items.length ?? 0) === 0 ? (
        <EmptyState
          title="No training runs yet"
          body="Training needs a valid dataset."
          action={{ href: "/datasets", label: "Register dataset" }}
        />
      ) : (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Family</th>
                <th>Dataset</th>
                <th>Status</th>
                <th>Started</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {[...(runs.data?.items ?? [])]
                .sort((left, right) => right.created_at.localeCompare(left.created_at))
                .map((run) => (
                  <tr key={run.id}>
                    <td>
                      <button
                        type="button"
                        className="link-button mono"
                        title={run.id}
                        onClick={() => {
                          const query = new URLSearchParams(params.toString());
                          query.set("run", run.id);
                          router.replace(`${pathname}?${query.toString()}`);
                        }}
                      >
                        {formatId(run.id)}
                      </button>
                    </td>
                    <td>{familyLabel(run.model_family)}</td>
                    <td className="mono">{run.dataset_version}</td>
                    <td>
                      <StatusBadge kind="training" value={run.status} />
                    </td>
                    <td>{formatDate(run.started_at ?? run.created_at)}</td>
                    <td>{duration(run.started_at, run.finished_at)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
      {selected ? (
        <section className="card">
          <h2>Run {formatId(selected.id)}</h2>
          <JobStepper status={selected.status} errorMessage={selected.error_message} />
          <details>
            <summary>Configuration</summary>
            <pre className="mono">{JSON.stringify(selected.configuration, null, 2)}</pre>
          </details>
          {selected.metrics ? (
            <p>Global WAPE {formatRatioPercent(selected.metrics.metrics.global.wape, 1)}</p>
          ) : null}
          {models.data?.items
            .filter((model) => model.training_run_id === selected.id)
            .map((model) => (
              <p key={model.id}>
                <Link href={`/models?model=${model.id}`}>Open model {model.version}</Link>
              </p>
            ))}
        </section>
      ) : null}
      <Dialog title="Start training" open={open} onClose={() => setOpen(false)}>
        <label className="field">
          Dataset
          <select value={datasetId || valid[0]?.id || ""} onChange={(event) => setDatasetId(event.target.value)}>
            {valid.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.name} v{dataset.version}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Model family
          <select value={family} onChange={(event) => setFamily(modelFamilySchema.parse(event.target.value))}>
            {FAMILIES.map((item) => (
              <option key={item} value={item}>
                {familyLabel(item)}
              </option>
            ))}
          </select>
        </label>
        <p className="caption">Defaults: 3 backtest folds and a 7-day horizon.</p>
        <p className="caption">
          {remaining == null
            ? "Daily training limit is still loading."
            : `${remaining} training runs remaining today.`}
        </p>
        <div className="dialog-actions">
          <button type="button" className="btn btn-secondary" data-cancel onClick={() => setOpen(false)}>
            Cancel
          </button>
          <button type="button" className="btn btn-primary" disabled={start.isPending} onClick={() => void submit()}>
            {start.isPending ? "Starting…" : "Start training"}
          </button>
        </div>
      </Dialog>
    </main>
  );
}

function duration(started: string | null, finished: string | null): string {
  if (!started || !finished) {
    return MISSING_MARK;
  }
  const seconds = (new Date(finished).getTime() - new Date(started).getTime()) / 1000;
  return formatDuration(seconds);
}

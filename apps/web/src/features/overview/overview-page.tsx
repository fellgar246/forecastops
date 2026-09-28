"use client";

import Link from "next/link";
import { Check } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ErrorNotice } from "@/components/error-notice";
import { ForecastChart } from "@/components/forecast-chart";
import { KpiCard } from "@/components/kpi-card";
import { MissingValue } from "@/components/missing-value";
import { ContextBar, PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { GenerateForecastButton } from "@/features/actions/generate-forecast-dialog";
import { useDatasets, useForecastSeries, useForecasts, useModelPerformance, useModels, useTrainingRuns } from "@/lib/api/queries";
import { chartPoints } from "@/lib/chart-points";
import { formatDate, formatHorizon, formatRatioPercent, formatRelative, formatUnits } from "@/lib/format";
import { familyLabel, readToken } from "@/lib/status";

export function OverviewPage() {
  const models = useModels();
  const training = useTrainingRuns();
  const forecasts = useForecasts();
  const performance = useModelPerformance();
  const datasets = useDatasets();
  const error = models.error ?? training.error ?? forecasts.error ?? performance.error ?? datasets.error;
  const loading =
    models.isPending || training.isPending || forecasts.isPending || performance.isPending || datasets.isPending;
  const production = models.data?.items.find((model) => model.status === "PRODUCTION");
  const succeeded = (forecasts.data?.items ?? [])
    .filter((item) => item.status === "SUCCEEDED")
    .sort((left, right) => right.created_at.localeCompare(left.created_at));
  const owned = production ? succeeded.filter((item) => item.model_id === production.id) : [];
  const forecast = owned[0] ?? succeeded[0];
  const series = useForecastSeries(forecast?.id, {});
  const lastRun = training.data?.items.find((run) => run.id === production?.training_run_id);
  const ready = Boolean(forecast);
  if (loading) {
    return (
      <main className="page">
        <PageHeader title="Overview" description="Production forecast and model health at a glance." />
        <p className="skeleton-block" aria-hidden="true" />
      </main>
    );
  }
  if (error) {
    return (
      <main className="page">
        <PageHeader title="Overview" description="Production forecast and model health at a glance." />
        <ErrorNotice error={error} onRetry={() => void models.refetch()} />
      </main>
    );
  }
  return (
    <main className="page">
      <PageHeader
        title="Overview"
        description="Production forecast and model health at a glance."
        actions={<GenerateForecastButton />}
      />
      <ContextBar
        items={[
          {
            label: "Dataset",
            value: forecast ? `v${forecast.dataset_version}` : <MissingValue reason="No forecast has run yet." />,
          },
          {
            label: "Model",
            value: production ? (
              `${familyLabel(production.model_family)} ${production.version}`
            ) : (
              <MissingValue reason="No production model yet." />
            ),
          },
          {
            label: "Updated",
            value: forecast ? (
              <span title={formatDate(forecast.created_at)}>{formatRelative(forecast.created_at)}</span>
            ) : (
              <MissingValue reason="No forecast has run yet." />
            ),
          },
        ]}
      />
      {forecast && series.data ? (
        <Snapshot
          production={production}
          lastRun={lastRun}
          forecast={forecast}
          series={series.data}
          models={performance.data?.items ?? []}
        />
      ) : null}
      {ready && series.isPending ? <p className="skeleton-block" aria-hidden="true" /> : null}
      {ready ? null : (
        <GettingStarted
          datasets={datasets.data?.items ?? []}
          models={models.data?.items ?? []}
          training={training.data?.items ?? []}
          forecasts={forecasts.data?.items ?? []}
        />
      )}
      {series.error ? <ErrorNotice error={series.error} onRetry={() => void series.refetch()} /> : null}
    </main>
  );
}

function Snapshot({
  production,
  lastRun,
  forecast,
  series,
  models,
}: {
  production: { model_family: string; version: string; metrics: { wape: number } | null } | undefined;
  lastRun: { status: string; finished_at: string | null } | undefined;
  forecast: { horizon: number; granularity: "day" | "week"; model_version: string; dataset_version: string };
  series: NonNullable<ReturnType<typeof useForecastSeries>["data"]>;
  models: { id: string; model_family: string; version: string; created_at: string; status: string; metrics: { wape: number } | null }[];
}) {
  const points = chartPoints(series);
  const range =
    series.p10_total == null || series.p90_total == null
      ? "No uncertainty range is available."
      : `${formatUnits(series.p10_total)} – ${formatUnits(series.p90_total)}`;
  const actuals = series.daily.some((point) => point.actual != null);
  return (
    <>
      <section className="kpi-grid" aria-label="Production snapshot">
        <KpiCard
          label="Active model"
          value={production ? `${familyLabel(production.model_family)} ${production.version}` : <MissingValue reason="No production model yet." />}
          caption={production ? "Production" : <Link href="/models">Review models</Link>}
        />
        <KpiCard
          label="Last training run"
          value={lastRun ? <StatusBadge kind="training" value={lastRun.status} size="md" /> : <MissingValue reason="No training run is linked to the production model." />}
          caption={lastRun?.finished_at ? formatDate(lastRun.finished_at) : "Waiting for a finished run"}
        />
        <KpiCard
          label="Global WAPE"
          term="WAPE"
          value={formatRatioPercent(production?.metrics?.wape, 1)}
          caption={production ? `${familyLabel(production.model_family)} ${production.version}` : "No production model yet"}
        />
        <KpiCard
          label="Forecast horizon"
          term="Horizon"
          value={formatHorizon(forecast.horizon, forecast.granularity)}
          caption={`Dataset v${forecast.dataset_version}`}
        />
        <KpiCard
          label="Last data update"
          value={forecast ? `v${forecast.dataset_version}` : <MissingValue reason="No dataset is attached to this forecast." />}
          caption="Dataset version used for this forecast"
        />
        <KpiCard
          label="Forecasted demand"
          term="P50"
          value={series.p50_total == null ? <MissingValue reason="This forecast has no P50 total." /> : formatUnits(series.p50_total)}
          caption={range}
        />
      </section>
      <ForecastChart
        points={points}
        cutoff={series.cutoff}
        subtitle={`${forecast.granularity === "week" ? "Weekly" : "Daily"} units · ${formatHorizon(forecast.horizon, forecast.granularity)} horizon`}
        modelVersion={forecast.model_version}
      />
      <div className="split-2">
        <section className="card">
          <h2>Actual vs forecast</h2>
          {actuals ? (
            <p>Actuals on this forecast are shown as dots on the demand chart.</p>
          ) : (
            <p>Actuals after this forecast was made are not available yet.</p>
          )}
        </section>
        <WapeTrend models={models} />
      </div>
    </>
  );
}

function WapeTrend({
  models,
}: {
  models: { model_family: string; version: string; created_at: string; status: string; metrics: { wape: number } | null }[];
}) {
  const points = models
    .filter((model) => model.metrics != null)
    .map((model) => ({
      date: model.created_at,
      wape: model.metrics?.wape ?? null,
      label: `${familyLabel(model.model_family)} ${model.version}`,
    }));
  const production = models.find((model) => model.status === "PRODUCTION")?.metrics?.wape;
  if (points.length === 0) {
    return (
      <section className="card">
        <h2>WAPE trend</h2>
        <p>No evaluation scores are stored yet.</p>
      </section>
    );
  }
  return (
    <section className="card">
      <h2>WAPE trend</h2>
      <p className="caption">WAPE (lower is better)</p>
      <div className="chart-frame" role="img" aria-label="WAPE by model version">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={points}>
            <CartesianGrid stroke={readToken("--border")} vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 12, fill: readToken("--text-muted") }} />
            <YAxis tickFormatter={(value: number) => formatRatioPercent(value, 0)} tick={{ fontSize: 12, fill: readToken("--text-muted") }} />
            <Tooltip formatter={(value) => formatRatioPercent(typeof value === "number" ? value : null, 1)} />
            {production != null ? (
              <ReferenceLine y={production} stroke={readToken("--viz-threshold")} strokeDasharray="4 4" label="Production WAPE" />
            ) : null}
            <Line dataKey="wape" name="WAPE" stroke={readToken("--viz-p50")} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function GettingStarted({
  datasets,
  models,
  training,
  forecasts,
}: {
  datasets: { status: string }[];
  models: { status: string }[];
  training: { status: string }[];
  forecasts: { status: string }[];
}) {
  const steps = [
    { label: "Register a dataset", done: datasets.length > 0, href: "/datasets" },
    { label: "Validate the dataset", done: datasets.some((item) => item.status === "valid"), href: "/datasets" },
    { label: "Train a model", done: training.some((run) => run.status === "COMPLETED"), href: "/training" },
    {
      label: "Approve a model",
      done: models.some((model) => model.status === "APPROVED" || model.status === "PRODUCTION"),
      href: "/models",
    },
    { label: "Generate a forecast", done: forecasts.some((item) => item.status === "SUCCEEDED"), href: "/forecasts" },
  ];
  return (
    <section className="empty-state">
      <h2>No forecasts yet</h2>
      <p>The overview fills in once a model is approved and a forecast has run.</p>
      <ol className="checklist">
        {steps.map((step) => (
          <li key={step.label}>
            {step.done ? <Check aria-hidden="true" size={16} /> : <span className="checklist-open" aria-hidden="true" />}
            <Link href={step.href}>{step.label}</Link>
          </li>
        ))}
      </ol>
      <Link className="btn btn-primary" href="/datasets">
        Start with a dataset
      </Link>
    </section>
  );
}

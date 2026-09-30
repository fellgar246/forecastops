"use client";

import Link from "next/link";
import { DriversPanel } from "@/components/drivers-panel";
import { ErrorNotice } from "@/components/error-notice";
import { ExplanationPanel } from "@/components/explanation-panel";
import { ForecastChart } from "@/components/forecast-chart";
import { MissingValue } from "@/components/missing-value";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { useExplanation, useForecast, useForecastSeries, useModel, useRequestExplanation } from "@/lib/api/queries";
import { chartPoints } from "@/lib/chart-points";
import { formatDate, formatHorizon, formatId, formatRatioPercent, formatUnits } from "@/lib/format";
import { familyLabel } from "@/lib/status";

export function ForecastDetailPage({ id }: { id: string }) {
  const forecast = useForecast(id);
  const series = useForecastSeries(id, {});
  const model = useModel(forecast.data?.model_id);
  const explanation = useExplanation(id);
  const generate = useRequestExplanation(id);
  if (forecast.error) {
    return (
      <main className="page">
        <ErrorNotice error={forecast.error} onRetry={() => void forecast.refetch()} />
      </main>
    );
  }
  if (!forecast.data) {
    return (
      <main className="page">
        <p className="skeleton-block" aria-hidden="true" />
      </main>
    );
  }
  const run = forecast.data;
  const points = series.data ? chartPoints(series.data) : [];
  const coverage = model.data?.promotion?.p90_coverage ?? model.data?.metrics?.p90_coverage;
  return (
    <main className="page">
      <p className="breadcrumb">
        <Link href="/forecasts">Forecasts</Link>
        <span aria-hidden="true"> / </span>
        <span className="mono" title={run.id}>
          {formatId(run.id)}
        </span>
      </p>
      <PageHeader
        title="Forecast"
        description="Expected demand for this run, with the uncertainty range when the model provides one."
      />
      <div className="split-detail">
        <div>
          {series.data ? (
            <ForecastChart
              points={points}
              cutoff={series.data.cutoff}
              subtitle={`${run.granularity === "week" ? "Weekly" : "Daily"} units · ${formatHorizon(run.horizon, run.granularity)} horizon`}
              modelVersion={run.model_version}
            />
          ) : null}
          {series.error ? <ErrorNotice error={series.error} onRetry={() => void series.refetch()} /> : null}
          {series.data ? (
            <details className="card">
              <summary>Points</summary>
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Series</th>
                      <th>Date</th>
                      <th className="numeric">P50</th>
                      <th className="numeric">P10</th>
                      <th className="numeric">P90</th>
                    </tr>
                  </thead>
                  <tbody>
                    {series.data.items.slice(0, 200).map((point) => (
                      <tr key={`${point.series_id}-${point.date}`}>
                        <td className="mono">{point.series_id}</td>
                        <td>{formatDate(point.date)}</td>
                        <td className="numeric">{formatUnits(point.p50)}</td>
                        <td className="numeric">{formatUnits(point.p10)}</td>
                        <td className="numeric">{formatUnits(point.p90)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          ) : null}
        </div>
        <aside className="stack">
          <section className="card">
            <h2>Run</h2>
            <p>
              <StatusBadge kind="forecast" value={run.status} size="md" />
            </p>
            <dl className="meta-list">
              <div>
                <dt>Model</dt>
                <dd>{familyLabel(run.model_family)}</dd>
              </div>
              <div>
                <dt>Model version</dt>
                <dd className="mono">{run.model_version}</dd>
              </div>
              <div>
                <dt>Dataset version</dt>
                <dd className="mono">{run.dataset_version}</dd>
              </div>
              <div>
                <dt>Horizon</dt>
                <dd>{formatHorizon(run.horizon, run.granularity)}</dd>
              </div>
              <div>
                <dt>Granularity</dt>
                <dd>{run.granularity}</dd>
              </div>
              <div>
                <dt>Created</dt>
                <dd>{formatDate(run.created_at)}</dd>
              </div>
            </dl>
            {run.error_message ? <p className="notice notice-danger">{run.error_message}</p> : null}
          </section>
          <section className="card">
            <h2>Uncertainty</h2>
            <p>
              P50 total{" "}
              {series.data?.p50_total == null ? (
                <MissingValue reason="This forecast has no P50 total." />
              ) : (
                formatUnits(series.data.p50_total)
              )}
            </p>
            <p>
              P10–P90{" "}
              {series.data?.p10_total == null || series.data?.p90_total == null ? (
                <MissingValue reason="This model does not produce quantiles, so the range does not apply." />
              ) : (
                `${formatUnits(series.data.p10_total)} – ${formatUnits(series.data.p90_total)}`
              )}
            </p>
            <p>
              P90 coverage{" "}
              {coverage == null ? (
                <MissingValue reason="This model does not produce quantiles, so P90 coverage does not apply." />
              ) : (
                formatRatioPercent(coverage, 0)
              )}
            </p>
          </section>
          <DriversPanel
            family={run.model_family}
            signals={explanation.data?.kind === "ready" ? explanation.data.explanation.signals : []}
          />
          <ExplanationPanel
            view={generate.isPending ? { kind: "pending" } : explanation.data}
            onGenerate={run.status === "SUCCEEDED" ? () => void generate.mutate() : undefined}
          />
          {explanation.error ? <ErrorNotice error={explanation.error} onRetry={() => void explanation.refetch()} /> : null}
          {generate.error ? <ErrorNotice error={generate.error} onRetry={() => void generate.mutate()} /> : null}
        </aside>
      </div>
    </main>
  );
}

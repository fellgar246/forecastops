"use client";

import type { ReactNode } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { EmptyState } from "@/components/empty-state";
import { ErrorNotice } from "@/components/error-notice";
import { MissingValue } from "@/components/missing-value";
import { PageHeader } from "@/components/page-header";
import { useModelPerformance, useTrainingRuns } from "@/lib/api/queries";
import type { ModelVersion } from "@/lib/api/schemas";
import { formatBias, formatDecimal, formatRatioPercent } from "@/lib/format";
import { familyLabel, readToken } from "@/lib/status";

export function ComparePage() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const performance = useModelPerformance();
  const training = useTrainingRuns();
  const models = (performance.data?.items ?? []).filter((model) => model.metrics != null);
  const versions = [...new Set(models.map((model) => model.dataset_version))];
  const selected = params.get("dataset") ?? versions[0] ?? "";
  const rows = models
    .filter((model) => model.dataset_version === selected)
    .sort((left, right) => (left.metrics?.wape ?? 0) - (right.metrics?.wape ?? 0));
  if (performance.isPending) {
    return (
      <main className="page">
        <PageHeader title="Compare" description="Model families scored on one dataset version." />
        <p className="skeleton-block" aria-hidden="true" />
      </main>
    );
  }
  if (performance.error) {
    return (
      <main className="page">
        <PageHeader title="Compare" description="Model families scored on one dataset version." />
        <ErrorNotice error={performance.error} onRetry={() => void performance.refetch()} />
      </main>
    );
  }
  return (
    <main className="page">
      <PageHeader
        title="Compare"
        description="Comparisons are only valid on one dataset version."
      />
      {versions.length > 0 ? (
        <label className="field">
          Dataset version
          <select
            aria-label="Dataset version"
            value={selected}
            onChange={(event) => {
              const query = new URLSearchParams(params.toString());
              query.set("dataset", event.target.value);
              router.replace(`${pathname}?${query.toString()}`);
            }}
          >
            {versions.map((version) => (
              <option key={version} value={version}>
                {version}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {models.length < 2 ? (
        <EmptyState
          title="Nothing to compare yet"
          body="Train at least two model families on the same dataset version."
          action={{ href: "/training", label: "Start training" }}
        />
      ) : (
        <>
          <section className="card">
            <h2>WAPE by family</h2>
            <div className="chart-frame" role="img" aria-label="WAPE by model family, lowest first">
              <ResponsiveContainer width="100%" height={Math.max(180, rows.length * 48)}>
                <BarChart data={rows.map(barRow)} layout="vertical" margin={{ left: 24, right: 48 }}>
                  <CartesianGrid stroke={readToken("--border")} horizontal={false} />
                  <XAxis type="number" tickFormatter={(value: number) => formatRatioPercent(value, 0)} />
                  <YAxis type="category" dataKey="label" width={160} />
                  <Tooltip formatter={(value) => formatRatioPercent(typeof value === "number" ? value : null, 1)} />
                  <Bar dataKey="wape" fill={readToken("--viz-p50")}>
                    <LabelList dataKey="wapeLabel" position="right" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
          <CompareTable rows={rows} training={training.data?.items ?? []} />
        </>
      )}
    </main>
  );
}

function barRow(model: ModelVersion) {
  return {
    label: `${familyLabel(model.model_family)} ${model.version}`,
    wape: model.metrics?.wape ?? 0,
    wapeLabel: formatRatioPercent(model.metrics?.wape, 1),
    family: model.model_family,
  };
}

function CompareTable({
  rows,
  training,
}: {
  rows: ModelVersion[];
  training: { id: string; metrics: { fold_count: number } | null }[];
}) {
  const bestWape = min(rows.map((row) => row.metrics?.wape));
  const bestSmape = min(rows.map((row) => row.metrics?.smape));
  const bestMae = min(rows.map((row) => row.metrics?.mae));
  const bestRmse = min(rows.map((row) => row.metrics?.rmse));
  const bestBias = min(rows.map((row) => (row.metrics ? Math.abs(row.metrics.bias) : null)));
  const bestCoverage = max(rows.map((row) => row.metrics?.p90_coverage ?? row.promotion?.p90_coverage ?? null));
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>Family</th>
            <th>Version</th>
            <th className="numeric">WAPE</th>
            <th className="numeric">sMAPE</th>
            <th className="numeric">MAE</th>
            <th className="numeric">RMSE</th>
            <th className="numeric">Bias</th>
            <th className="numeric">P90 coverage</th>
            <th className="numeric">Folds</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const folds = training.find((run) => run.id === row.training_run_id)?.metrics?.fold_count;
            const coverage = row.metrics?.p90_coverage ?? row.promotion?.p90_coverage ?? null;
            return (
              <tr key={row.id}>
                <td>
                  <span className="family-dot" data-family={row.model_family} />
                  {familyLabel(row.model_family)}
                </td>
                <td className="mono">{row.version}</td>
                <MetricCell value={formatRatioPercent(row.metrics?.wape, 1)} best={row.metrics?.wape === bestWape} />
                <MetricCell value={formatRatioPercent(row.metrics?.smape, 1)} best={row.metrics?.smape === bestSmape} />
                <MetricCell value={formatDecimal(row.metrics?.mae, 1)} best={row.metrics?.mae === bestMae} />
                <MetricCell value={formatDecimal(row.metrics?.rmse, 1)} best={row.metrics?.rmse === bestRmse} />
                <MetricCell
                  value={formatBias(row.metrics?.bias)}
                  best={row.metrics != null && Math.abs(row.metrics.bias) === bestBias}
                />
                <MetricCell
                  value={
                    coverage == null ? (
                      <MissingValue reason="This model does not produce quantiles, so P90 coverage does not apply." />
                    ) : (
                      formatRatioPercent(coverage, 0)
                    )
                  }
                  best={coverage != null && coverage === bestCoverage}
                />
                <td className="numeric">
                  {folds == null ? <MissingValue reason="Fold count is not stored on this run." /> : folds}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function MetricCell({ value, best }: { value: ReactNode; best: boolean }) {
  return (
    <td className="numeric">
      {value}
      {best ? <span className="best-tag">best</span> : null}
    </td>
  );
}

function min(values: (number | null | undefined)[]): number | undefined {
  const present = values.filter((value): value is number => value != null);
  return present.length === 0 ? undefined : Math.min(...present);
}

function max(values: (number | null | undefined)[]): number | undefined {
  const present = values.filter((value): value is number => value != null);
  return present.length === 0 ? undefined : Math.max(...present);
}

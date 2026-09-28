"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useMemo } from "react";
import { EmptyState } from "@/components/empty-state";
import { ErrorNotice } from "@/components/error-notice";
import { FilterBar, type ForecastFilters } from "@/components/filter-bar";
import { MissingValue } from "@/components/missing-value";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { GenerateForecastButton } from "@/features/actions/generate-forecast-dialog";
import { useCatalog, useForecastSeries, useForecasts, useHealth } from "@/lib/api/queries";
import { formatDate, formatId, formatUnits } from "@/lib/format";

export function ForecastsPage() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const filters = useMemo<ForecastFilters>(
    () => ({
      category: params.get("category") ?? "",
      sku: params.get("sku") ?? "",
      store: params.get("store") ?? "",
      horizon: params.get("horizon") ?? "",
    }),
    [params],
  );
  const onChange = useCallback(
    (next: ForecastFilters) => {
      const query = new URLSearchParams();
      if (next.category) query.set("category", next.category);
      if (next.sku) query.set("sku", next.sku);
      if (next.store) query.set("store", next.store);
      if (next.horizon) query.set("horizon", next.horizon);
      const text = query.toString();
      router.replace(text ? `${pathname}?${text}` : pathname);
    },
    [pathname, router],
  );
  const forecasts = useForecasts();
  const health = useHealth();
  const newest = [...(forecasts.data?.items ?? [])].sort((left, right) =>
    right.created_at.localeCompare(left.created_at),
  );
  const horizonNumber = filters.horizon ? Number(filters.horizon) : undefined;
  const matched = horizonNumber == null ? newest : newest.filter((item) => item.horizon === horizonNumber);
  const target = matched[0] ?? newest[0];
  const series = useForecastSeries(target?.id, {
    category: filters.category || undefined,
    sku: filters.sku || undefined,
    store: filters.store || undefined,
    horizon: horizonNumber,
  });
  const catalog = useCatalog(target?.dataset_id);
  const horizons = [...new Set(newest.map((item) => item.horizon))]
    .filter((value) => health.data == null || value <= health.data.max_forecast_horizon_days)
    .sort((left, right) => left - right);
  const filtered = Boolean(filters.category || filters.sku || filters.store || filters.horizon);
  if (forecasts.isPending) {
    return (
      <main className="page">
        <PageHeader title="Forecasts" description="Demand ranges for each series in a forecast run." />
        <p className="skeleton-block" aria-hidden="true" />
      </main>
    );
  }
  if (forecasts.error) {
    return (
      <main className="page">
        <PageHeader title="Forecasts" description="Demand ranges for each series in a forecast run." />
        <ErrorNotice error={forecasts.error} onRetry={() => void forecasts.refetch()} />
      </main>
    );
  }
  return (
    <main className="page">
      <PageHeader
        title="Forecasts"
        description="Demand ranges for each series in a forecast run."
        actions={<GenerateForecastButton />}
      />
      <FilterBar catalog={catalog.data} horizons={horizons} values={filters} onChange={onChange} />
      {newest.length === 0 ? (
        <EmptyState
          title="No forecasts to show"
          body="Generate a forecast from an approved model to see demand ranges here."
          action={{ href: "/models", label: "Review models" }}
        />
      ) : null}
      {newest.length > 0 && series.data && series.data.series.length === 0 ? (
        <EmptyState
          title={filtered ? "No series match these filters" : "No series to show"}
          body={
            filtered
              ? "Try a different category, store, or horizon."
              : "This forecast run has no stored points."
          }
          action={filtered ? undefined : { href: "/models", label: "Review models" }}
        />
      ) : null}
      {filtered && series.data?.series.length === 0 ? (
        <button type="button" className="btn btn-secondary" onClick={() => onChange({ category: "", sku: "", store: "", horizon: "" })}>
          Clear filters
        </button>
      ) : null}
      {series.error ? <ErrorNotice error={series.error} onRetry={() => void series.refetch()} /> : null}
      {target && series.data && series.data.series.length > 0 ? (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Series</th>
                <th className="numeric">Horizon</th>
                <th className="numeric">P50 total</th>
                <th className="numeric">P10–P90</th>
                <th>Model version</th>
                <th>Created</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {series.data.series.map((item) => (
                <tr key={item.series_id}>
                  <td>
                    <Link className="mono" href={`/forecasts/${target.id}`} title={item.series_id}>
                      {formatId(item.series_id)}
                    </Link>
                  </td>
                  <td className="numeric">{target.horizon}</td>
                  <td className="numeric">
                    {item.p50_total == null ? <MissingValue reason="No P50 total for this series." /> : formatUnits(item.p50_total)}
                  </td>
                  <td className="numeric">
                    {item.p10_total == null || item.p90_total == null ? (
                      <MissingValue reason="This model does not produce quantiles, so the range does not apply." />
                    ) : (
                      `${formatUnits(item.p10_total)} – ${formatUnits(item.p90_total)}`
                    )}
                  </td>
                  <td className="mono">{target.model_version}</td>
                  <td>{formatDate(target.created_at)}</td>
                  <td>
                    <StatusBadge kind="forecast" value={target.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </main>
  );
}

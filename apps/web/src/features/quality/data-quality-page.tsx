"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { EmptyState } from "@/components/empty-state";
import { ErrorNotice } from "@/components/error-notice";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { useDataQuality, useDatasets } from "@/lib/api/queries";
import type { QualityReport } from "@/lib/api/schemas";
import { formatRatioPercent } from "@/lib/format";

const RULES: Record<string, string> = {
  missing_required: "Missing values",
  duplicate_grain: "Duplicates",
  calendar_gap: "Date gaps",
  negative_units_sold: "Negative demand",
  stockout_rate: "Stock-outs",
  unexpected_category: "Unexpected categories",
  stale_dataset: "Stale data",
  negative_price: "Negative price",
  invalid_promotion_discount: "Invalid promotion discount",
  unknown_store: "Unknown store",
  unknown_sku: "Unknown SKU",
  extreme_outlier: "Extreme outlier",
  zero_demand_fraction: "Zero demand",
};

export function DataQualityPage() {
  const quality = useDataQuality();
  const datasets = useDatasets();
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const items = quality.data?.items ?? [];
  const selectedId = params.get("dataset") ?? items[0]?.dataset_id ?? "";
  const selected = items.find((item) => item.dataset_id === selectedId) ?? items[0];
  const dataset = datasets.data?.items.find((item) => item.id === selected?.dataset_id);
  if (quality.isPending || datasets.isPending) {
    return (
      <main className="page">
        <PageHeader title="Data quality" description="What would block training, and what is only a warning." />
        <p className="skeleton-block" aria-hidden="true" />
      </main>
    );
  }
  if (quality.error) {
    return (
      <main className="page">
        <PageHeader title="Data quality" description="What would block training, and what is only a warning." />
        <ErrorNotice error={quality.error} onRetry={() => void quality.refetch()} />
      </main>
    );
  }
  return (
    <main className="page">
      <PageHeader
        title="Data quality"
        description="What would block training, and what is only a warning."
        actions={
          dataset ? (
            <a className="btn btn-secondary" href="/datasets">
              Validate dataset
            </a>
          ) : null
        }
      />
      {items.length > 0 ? (
        <label className="field">
          Dataset version
          <select
            aria-label="Dataset version"
            value={selected?.dataset_id ?? ""}
            onChange={(event) => {
              const query = new URLSearchParams(params.toString());
              query.set("dataset", event.target.value);
              router.replace(`${pathname}?${query.toString()}`);
            }}
          >
            {items.map((item) => (
              <option key={item.dataset_id} value={item.dataset_id}>
                {item.dataset_version}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {!selected || !selected.quality_report ? (
        <EmptyState
          title="No quality report yet"
          body="Validate a dataset to see missing values, duplicates, gaps, and stock-outs."
          action={{ href: "/datasets", label: "Validate dataset" }}
        />
      ) : (
        <Report report={selected.quality_report} version={selected.dataset_version} dateMax={dataset?.date_max ?? null} />
      )}
    </main>
  );
}

function Report({
  report,
  version,
  dateMax,
}: {
  report: NonNullable<QualityReport>;
  version: string;
  dateMax: string | null;
}) {
  const findings = [
    ...report.blocking.map((item) => ({ ...item, severity: "blocking" as const })),
    ...report.advisory.map((item) => ({ ...item, severity: "advisory" as const })),
  ];
  return (
    <>
      <p className="caption">
        Dataset {version}
        {dateMax ? ` · last observation ${dateMax}` : ""}
      </p>
      <section className="kpi-grid">
        <article className="kpi-card">
          <p className="overline">Overall</p>
          <p className="kpi-value">
            <StatusBadge kind="dataset" value={report.status} size="md" />
          </p>
        </article>
        <article className="kpi-card">
          <p className="overline">Blocking findings</p>
          <p className="kpi-value">{report.blocking.length}</p>
        </article>
        <article className="kpi-card">
          <p className="overline">Advisory findings</p>
          <p className="kpi-value">{report.advisory.length}</p>
        </article>
        <article className="kpi-card">
          <p className="overline">Stock-out rate</p>
          <p className="kpi-value">{formatRatioPercent(report.rates.stockouts, 1)}</p>
        </article>
        <article className="kpi-card">
          <p className="overline">Missing values</p>
          <p className="kpi-value">{formatRatioPercent(report.rates.missing_values, 1)}</p>
        </article>
        <article className="kpi-card">
          <p className="overline">Duplicates</p>
          <p className="kpi-value">{formatRatioPercent(report.rates.duplicates, 1)}</p>
        </article>
      </section>
      {findings.length === 0 ? <p>No findings. The checks found nothing to report.</p> : null}
      <ul className="finding-list">
        {findings.map((finding) => (
          <li key={`${finding.severity}-${finding.code}`} className={finding.severity === "blocking" ? "notice notice-danger" : "notice notice-warning"}>
            <strong>{RULES[finding.code] ?? finding.message}</strong>
            <p>
              {finding.count} rows
              {finding.rate == null ? "" : ` · ${formatRatioPercent(finding.rate, 1)}`}
            </p>
            <p>{finding.message}</p>
          </li>
        ))}
      </ul>
    </>
  );
}

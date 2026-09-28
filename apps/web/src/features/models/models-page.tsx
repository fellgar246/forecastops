"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useMemo, useState } from "react";
import { EmptyState } from "@/components/empty-state";
import { ErrorNotice } from "@/components/error-notice";
import { MissingValue } from "@/components/missing-value";
import { PageHeader } from "@/components/page-header";
import { QualityGatePanel } from "@/components/quality-gate-panel";
import { ReviewActions } from "@/components/review-actions";
import { StatusBadge } from "@/components/status-badge";
import { useModels } from "@/lib/api/queries";
import type { ModelVersion } from "@/lib/api/schemas";
import { formatBias, formatDate, formatRatioPercent } from "@/lib/format";
import { familyLabel } from "@/lib/status";

const TABS = [
  { id: "all", label: "All" },
  { id: "pending", label: "Needs review" },
  { id: "production", label: "Production" },
  { id: "rejected", label: "Rejected" },
] as const;

export function ModelsPage() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const tab = params.get("tab") ?? "all";
  const [selectedId, setSelectedId] = useState<string | null>(params.get("model"));
  const models = useModels();
  const [sort, setSort] = useState<"created" | "wape">("created");
  const items = useMemo(() => {
    const source = models.data?.items ?? [];
    const filtered = source.filter((model) => {
      if (tab === "pending") return model.status === "PENDING_APPROVAL";
      if (tab === "production") return model.status === "PRODUCTION";
      if (tab === "rejected") return model.status === "REJECTED";
      return true;
    });
    return [...filtered].sort((left, right) => {
      if (sort === "wape") {
        return (left.metrics?.wape ?? Number.POSITIVE_INFINITY) - (right.metrics?.wape ?? Number.POSITIVE_INFINITY);
      }
      return right.created_at.localeCompare(left.created_at);
    });
  }, [models.data, sort, tab]);
  const selected = (models.data?.items ?? []).find((model) => model.id === selectedId);
  const production = (models.data?.items ?? []).find(
    (model) => model.status === "PRODUCTION" && model.model_family === selected?.model_family,
  );
  const pendingCount = (models.data?.items ?? []).filter((model) => model.status === "PENDING_APPROVAL").length;
  function setQuery(next: { tab?: string; model?: string | null }) {
    const query = new URLSearchParams(params.toString());
    if (next.tab) query.set("tab", next.tab);
    if (next.model === null) {
      query.delete("model");
      setSelectedId(null);
    } else if (next.model) {
      query.set("model", next.model);
      setSelectedId(next.model);
    }
    const text = query.toString();
    router.replace(text ? `${pathname}?${text}` : pathname);
  }
  if (models.isPending) {
    return (
      <main className="page">
        <PageHeader title="Models" description="Registered versions, scores, and the decision waiting on a person." />
        <p className="skeleton-block" aria-hidden="true" />
      </main>
    );
  }
  if (models.error) {
    return (
      <main className="page">
        <PageHeader title="Models" description="Registered versions, scores, and the decision waiting on a person." />
        <ErrorNotice error={models.error} onRetry={() => void models.refetch()} />
      </main>
    );
  }
  return (
    <main className="page">
      <PageHeader
        title="Models"
        description="Registered versions, scores, and the decision waiting on a person."
        actions={
          pendingCount > 0 ? (
            <button type="button" className="btn btn-primary" onClick={() => setQuery({ tab: "pending" })}>
              Review pending model
            </button>
          ) : null
        }
      />
      <div className="tabs" role="tablist">
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            className={tab === item.id ? "tab tab-active" : "tab"}
            onClick={() => setQuery({ tab: item.id })}
          >
            {item.label}
            {item.id === "pending" && pendingCount > 0 ? ` (${pendingCount})` : ""}
          </button>
        ))}
      </div>
      {models.data && models.data.items.length === 0 ? (
        <EmptyState
          title="No models yet"
          body="Models appear here after a training run completes and is evaluated."
          action={{ href: "/training", label: "Start training" }}
        />
      ) : null}
      {items.length > 0 ? (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Version</th>
                <th>Family</th>
                <th>Dataset</th>
                <th className="numeric">
                  <button type="button" className="sort-button" onClick={() => setSort("wape")}>
                    WAPE
                  </button>
                </th>
                <th className="numeric">Bias</th>
                <th>Status</th>
                <th>
                  <button type="button" className="sort-button" onClick={() => setSort("created")}>
                    Created
                  </button>
                </th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {items.map((model) => (
                <tr key={model.id}>
                  <td className="mono">{model.version}</td>
                  <td>
                    <span className="family-dot" data-family={model.model_family} />
                    {familyLabel(model.model_family)}
                  </td>
                  <td className="mono">{model.dataset_version}</td>
                  <td className="numeric">{formatRatioPercent(model.metrics?.wape, 1)}</td>
                  <td className="numeric">{formatBias(model.metrics?.bias)}</td>
                  <td>
                    <StatusBadge kind="model" value={model.status} reason={model.rejection_reason} />
                  </td>
                  <td>{formatDate(model.created_at)}</td>
                  <td>
                    <button type="button" className="btn btn-secondary" onClick={() => setQuery({ model: model.id })}>
                      {model.status === "PENDING_APPROVAL" ? "Review" : "Open"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">Showing 1–{Math.min(items.length, 25)} of {items.length}. Sorted by {sort === "created" ? "created time, newest first" : "WAPE, lowest first"}.</p>
        </div>
      ) : null}
      {selected ? (
        <ModelDrawer model={selected} production={production} onClose={() => setQuery({ model: null })} />
      ) : null}
    </main>
  );
}

function ModelDrawer({
  model,
  production,
  onClose,
}: {
  model: ModelVersion;
  production: ModelVersion | undefined;
  onClose: () => void;
}) {
  return (
    <div className="dialog-backdrop" onMouseDown={onClose}>
      <aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-drawer-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="card-header">
          <h2 id="model-drawer-title">
            {familyLabel(model.model_family)} {model.version}
          </h2>
          <button type="button" className="btn btn-ghost" onClick={onClose}>
            Close
          </button>
        </header>
        <StatusBadge kind="model" value={model.status} reason={model.rejection_reason} size="md" />
        <p>
          WAPE {model.metrics ? formatRatioPercent(model.metrics.wape, 1) : <MissingValue reason="No score is stored." />}
        </p>
        <QualityGatePanel model={model} />
        <ReviewActions model={model} production={production} />
      </aside>
    </div>
  );
}

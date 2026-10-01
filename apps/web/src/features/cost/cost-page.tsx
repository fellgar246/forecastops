"use client";

import { ErrorNotice } from "@/components/error-notice";
import { KpiCard } from "@/components/kpi-card";
import { MissingValue } from "@/components/missing-value";
import { PageHeader } from "@/components/page-header";
import { useCost } from "@/lib/api/queries";
import { formatCount, formatDateTime, formatMoney, formatTokens } from "@/lib/format";

const CAPABILITIES = [
  { key: "aws_enabled", label: "AWS" },
  { key: "aws_ml_enabled", label: "Cloud training and batch" },
  { key: "bedrock_enabled", label: "Bedrock" },
  { key: "sagemaker_enabled", label: "SageMaker" },
  { key: "training_enabled", label: "Training" },
  { key: "online_inference", label: "Online inference" },
] as const;

export function CostPage() {
  const cost = useCost();
  if (cost.isPending) {
    return (
      <main className="page">
        <PageHeader
          title="Cost"
          description="Monthly budget, usage, and which cloud capabilities are on."
        />
        <p className="skeleton-block" aria-hidden="true" />
      </main>
    );
  }
  if (cost.error || !cost.data) {
    return (
      <main className="page">
        <PageHeader
          title="Cost"
          description="Monthly budget, usage, and which cloud capabilities are on."
        />
        <ErrorNotice error={cost.error ?? new Error("Cost data is unavailable.")} onRetry={() => void cost.refetch()} />
      </main>
    );
  }
  const posture = cost.data;
  return (
    <main className="page">
      <PageHeader
        title="Cost"
        description="Monthly budget, usage, and which cloud capabilities are on."
      />
      <section className="kpi-grid" aria-label="Cost posture">
        <KpiCard
          label="Monthly budget"
          value={formatMoney(posture.monthly_budget_usd)}
          caption="USD this month"
        />
        <KpiCard
          label="Training runs"
          value={formatCount(posture.training_runs_this_month)}
          caption="This UTC month"
        />
        <KpiCard
          label="Explanation calls"
          value={formatCount(posture.explanation_calls_today)}
          caption="Today, UTC"
        />
        <KpiCard
          label="Explanation tokens"
          value={formatTokens(posture.approximate_explanation_tokens)}
          caption="Approximate total for today"
        />
      </section>
      <section className="card">
        <h2>Estimated spend</h2>
        <p>
          {posture.estimated_spend_usd == null ? (
            <MissingValue reason={posture.spend_note} />
          ) : (
            formatMoney(posture.estimated_spend_usd, true)
          )}
        </p>
        <p className="caption">{posture.spend_note}</p>
      </section>
      <section className="card">
        <h2>Cloud capabilities</h2>
        <dl className="meta-list">
          {CAPABILITIES.map((item) => (
            <div key={item.key}>
              <dt>{item.label}</dt>
              <dd>{posture[item.key] ? "On" : "Off"}</dd>
            </div>
          ))}
        </dl>
      </section>
      <section className="card">
        <h2>Last cleanup</h2>
        <p>
          {posture.last_cleanup_at ? (
            formatDateTime(posture.last_cleanup_at)
          ) : (
            <MissingValue reason="No cleanup has been recorded." />
          )}
        </p>
      </section>
    </main>
  );
}

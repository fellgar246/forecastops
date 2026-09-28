"use client";

import { ErrorNotice } from "@/components/error-notice";
import { MissingValue } from "@/components/missing-value";
import { PageHeader } from "@/components/page-header";
import { useHealth } from "@/lib/api/queries";

export function CostPage() {
  const health = useHealth();
  if (health.isPending) {
    return (
      <main className="page">
        <PageHeader title="Cost" description="Which spend limits apply in this environment." />
        <p className="skeleton-block" aria-hidden="true" />
      </main>
    );
  }
  if (health.error) {
    return (
      <main className="page">
        <PageHeader title="Cost" description="Which spend limits apply in this environment." />
        <ErrorNotice error={health.error} onRetry={() => void health.refetch()} />
      </main>
    );
  }
  const local = health.data?.execution_mode !== "aws";
  return (
    <main className="page">
      <PageHeader title="Cost" description="Which spend limits apply in this environment." />
      {local ? (
        <section className="notice notice-neutral" aria-labelledby="local-cost-title">
          <h2 id="local-cost-title">Local mode</h2>
          <p>Cloud spend tracking is inactive.</p>
        </section>
      ) : (
        <p>
          Billing data is not connected. <MissingValue reason="Billing data is not connected." />
        </p>
      )}
      <section className="card">
        <h2>Limits that still apply</h2>
        <dl className="meta-list">
          <div>
            <dt>Training runs per day</dt>
            <dd>{health.data?.max_training_jobs_per_day ?? <MissingValue reason="The training limit has not loaded." />}</dd>
          </div>
          <div>
            <dt>Forecast horizon</dt>
            <dd>
              {health.data
                ? `${health.data.max_forecast_horizon_days} days`
                : <MissingValue reason="The horizon limit has not loaded." />}
            </dd>
          </div>
        </dl>
      </section>
    </main>
  );
}

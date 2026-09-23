import { EnvironmentChip } from "@/components/environment-chip";

export default function HomePage() {
  return (
    <main className="page">
      <header>
        <p className="overline">ForecastOps</p>
        <h1>Demand forecasts for retail</h1>
        <p className="lede">
          Expected demand ranges for each product and store, produced in batch and reviewed by a
          person before production.
        </p>
      </header>
      <section className="context-bar" aria-label="Page context">
        <p>No dataset registered</p>
        <p>No production model</p>
      </section>
      <section className="empty-panel" aria-labelledby="local-profile-title">
        <h2 id="local-profile-title">Local profile</h2>
        <p>
          This environment runs on your machine. Cloud training, hosted inference, and
          language-model explanations are off.
        </p>
        <p>Forecasts, training, and model review are not available in this build yet.</p>
      </section>
      <EnvironmentChip />
    </main>
  );
}

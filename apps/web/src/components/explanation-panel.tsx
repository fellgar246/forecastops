import { Sparkles } from "lucide-react";
import type { Explanation, ExplanationView } from "@/lib/api/schemas";
import { formatDate } from "@/lib/format";

export function ExplanationPanel({
  view,
  failedCheck,
}: {
  view: ExplanationView | undefined;
  failedCheck?: string;
}) {
  return (
    <section className="card" aria-labelledby="explanation-title">
      <div className="card-header">
        <h2 id="explanation-title" className="explanation-title">
          <Sparkles aria-hidden="true" size={16} />
          Generated explanation
        </h2>
      </div>
      {view == null ? <p className="skeleton-line" /> : null}
      {view?.kind === "disabled" ? (
        <p className="notice notice-neutral">
          No explanation has been generated yet. Explanations are turned off in this environment.
        </p>
      ) : null}
      {view?.kind === "missing" ? (
        <p>No explanation has been generated yet.</p>
      ) : null}
      {view?.kind === "pending" ? (
        <p aria-live="polite">Generating explanation…</p>
      ) : null}
      {view?.kind === "invalid" ? (
        <p className="notice notice-warning">
          The generated explanation did not pass validation and is not shown.
          {failedCheck ? ` Failed check: ${failedCheck}.` : ` ${view.message}`}
        </p>
      ) : null}
      {view?.kind === "limited" ? <p className="notice notice-neutral">{view.message}</p> : null}
      {view?.kind === "ready" ? <Ready explanation={view.explanation} /> : null}
    </section>
  );
}

function Ready({ explanation }: { explanation: Explanation }) {
  return (
    <div>
      <p className="caption mono">
        {explanation.model_id} · {explanation.prompt_version}
      </p>
      <h3>Summary</h3>
      <p>{explanation.summary}</p>
      <h3>Signals</h3>
      <ul>
        {explanation.signals.map((signal) => (
          <li key={signal.name}>
            {signal.name} ({signal.direction})
          </li>
        ))}
      </ul>
      <h3>Risks</h3>
      <ul>
        {explanation.risks.map((risk) => (
          <li key={risk}>{risk}</li>
        ))}
      </ul>
      <h3>Uncertainty</h3>
      <p>{explanation.uncertainty}</p>
      <h3>Checks</h3>
      <ul>
        {explanation.checks.map((check) => (
          <li key={check}>{check}</li>
        ))}
      </ul>
      <p className="caption">
        Generated from the forecast package on {formatDate(explanation.generated_at)}. Numbers come
        from the model; the text is written by a language model and may be incomplete.
      </p>
    </div>
  );
}

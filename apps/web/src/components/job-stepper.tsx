import { StatusBadge } from "@/components/status-badge";

const STEPS = ["QUEUED", "PREPROCESSING", "TRAINING", "EVALUATING", "REGISTERING", "COMPLETED"] as const;

export function JobStepper({
  status,
  errorMessage,
}: {
  status: string;
  errorMessage: string | null;
}) {
  const failed = status === "FAILED";
  const current = failed ? -1 : STEPS.indexOf(status as (typeof STEPS)[number]);
  return (
    <div>
      <ol className="stepper" aria-live="polite">
        {STEPS.map((step, index) => {
          const state = failed ? "upcoming" : index < current ? "done" : index === current ? "current" : "upcoming";
          return (
            <li key={step} className={`step step-${state}`}>
              <StatusBadge kind="training" value={step} size="sm" />
            </li>
          );
        })}
        {failed ? (
          <li className="step step-current">
            <StatusBadge kind="training" value="FAILED" size="sm" />
          </li>
        ) : null}
      </ol>
      {failed && errorMessage ? <p className="notice notice-danger">{errorMessage}</p> : null}
    </div>
  );
}

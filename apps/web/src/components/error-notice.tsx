import { ApiError, ResponseParseError } from "@/lib/api/client";

export function ErrorNotice({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message =
    error instanceof Error ? error.message : "The request failed. Retry in a moment.";
  const code = error instanceof ApiError ? error.code : null;
  const retry = error instanceof ResponseParseError ? undefined : onRetry;
  return (
    <div className="notice notice-danger" role="alert">
      <p>{message}</p>
      {code ? (
        <details>
          <summary>Error code</summary>
          <p className="mono">{code}</p>
        </details>
      ) : null}
      {retry ? (
        <button type="button" className="btn btn-secondary" onClick={retry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

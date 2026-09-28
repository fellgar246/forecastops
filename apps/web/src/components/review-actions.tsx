"use client";

import { useState } from "react";
import { Dialog } from "@/components/dialog";
import { useToast } from "@/components/toast";
import { ApiError } from "@/lib/api/client";
import { useApproveModel, useRejectModel } from "@/lib/api/queries";
import type { ModelVersion } from "@/lib/api/schemas";
import { formatRatioPercent } from "@/lib/format";
import { familyLabel } from "@/lib/status";

export function ReviewActions({
  model,
  production,
}: {
  model: ModelVersion;
  production: ModelVersion | undefined;
}) {
  const toast = useToast();
  const approve = useApproveModel();
  const reject = useRejectModel();
  const [mode, setMode] = useState<"approve" | "promote" | "reject" | null>(null);
  const [reason, setReason] = useState("");
  const pending = model.status === "PENDING_APPROVAL";
  const busy = approve.isPending || reject.isPending;
  if (!pending) {
    return null;
  }
  async function confirm() {
    try {
      if (mode === "reject") {
        await reject.mutateAsync(model.id);
        toast("Model rejected");
      } else {
        await approve.mutateAsync({ id: model.id, promote: mode === "promote" });
        toast(mode === "promote" ? "Model promoted" : "Model approved");
      }
      setMode(null);
      setReason("");
    } catch (error) {
      const message = error instanceof ApiError ? error.message : "The decision could not be saved.";
      toast(message);
    }
  }
  const wape = model.metrics?.wape;
  return (
    <div className="review-actions">
      <button type="button" className="btn btn-primary" disabled={busy} onClick={() => setMode("approve")}>
        Approve
      </button>
      <button type="button" className="btn btn-secondary" disabled={busy} onClick={() => setMode("promote")}>
        Promote to production
      </button>
      <button type="button" className="btn btn-danger" disabled={busy} onClick={() => setMode("reject")}>
        Reject
      </button>
      <Dialog
        title={mode === "reject" ? "Reject model" : mode === "promote" ? "Promote to production" : "Approve model"}
        open={mode != null}
        onClose={() => setMode(null)}
      >
        <p>
          {familyLabel(model.model_family)} version {model.version} on dataset {model.dataset_version}. WAPE{" "}
          {formatRatioPercent(wape, 1)}.
        </p>
        {mode === "promote" ? (
          <p>
            {production
              ? `The current production model ${production.model_family} ${production.version} will return to approved.`
              : "This model will become the production model."}
          </p>
        ) : null}
        {mode === "reject" ? (
          <label className="field">
            Reason
            <textarea
              value={reason}
              onChange={(event) => setReason(event.target.value)}
              required
            />
            <span className="caption">A reviewer rejection is recorded as human.</span>
          </label>
        ) : null}
        <div className="dialog-actions">
          <button type="button" className="btn btn-secondary" data-cancel onClick={() => setMode(null)}>
            Cancel
          </button>
          <button
            type="button"
            className={mode === "reject" ? "btn btn-danger" : "btn btn-primary"}
            disabled={busy || (mode === "reject" && reason.trim() === "")}
            onClick={() => void confirm()}
          >
            {busy ? "Saving…" : mode === "reject" ? "Reject model" : mode === "promote" ? "Promote to production" : "Approve model"}
          </button>
        </div>
      </Dialog>
    </div>
  );
}

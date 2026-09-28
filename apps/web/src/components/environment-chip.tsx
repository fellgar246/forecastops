"use client";

import { useState } from "react";
import { useAwsHealth, useHealth } from "@/lib/api/queries";

export function EnvironmentChip() {
  const health = useHealth();
  const aws = useAwsHealth();
  const [open, setOpen] = useState(false);
  const mode = health.data?.execution_mode ?? "local";
  const label = mode === "local" ? "Local" : "Cloud";
  return (
    <div className="environment">
      <button
        type="button"
        className="environment-chip"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="environment-chip-label">Environment</span>
        {label}
      </button>
      {open ? (
        <div className="environment-popover" role="dialog" aria-label="Capabilities">
          <Capability name="Cloud ML" on={aws.data?.aws_ml_enabled ?? false} />
          <Capability name="Explanations" on={health.data?.bedrock_enabled ?? false} />
          <Capability name="Training" on={health.data?.training_enabled ?? false} />
          <Capability name="Hosted inference" on={aws.data?.sagemaker_enabled ?? false} />
        </div>
      ) : null}
    </div>
  );
}

function Capability({ name, on }: { name: string; on: boolean }) {
  return (
    <p>
      <span className={on ? "badge badge-sm badge-success" : "badge badge-sm badge-neutral"}>
        {on ? "On" : "Off"}
      </span>{" "}
      {name}
    </p>
  );
}

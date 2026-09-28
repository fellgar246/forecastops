import type { LucideIcon } from "lucide-react";
import {
  ChartColumn,
  Check,
  Circle,
  CircleCheck,
  CircleX,
  Clock,
  LoaderCircle,
  Star,
  UserRoundCheck,
  X,
} from "lucide-react";

export type StatusTone = "success" | "warning" | "danger" | "info" | "neutral";

export type StatusMeta = {
  label: string;
  tone: StatusTone;
  icon: LucideIcon;
  spin: boolean;
  solid: boolean;
};

const training: Record<string, StatusMeta> = {
  QUEUED: { label: "Queued", tone: "neutral", icon: Clock, spin: false, solid: false },
  PREPROCESSING: { label: "Preprocessing", tone: "info", icon: LoaderCircle, spin: true, solid: false },
  TRAINING: { label: "Training", tone: "info", icon: LoaderCircle, spin: true, solid: false },
  EVALUATING: { label: "Evaluating", tone: "info", icon: LoaderCircle, spin: true, solid: false },
  REGISTERING: { label: "Registering", tone: "info", icon: LoaderCircle, spin: true, solid: false },
  COMPLETED: { label: "Completed", tone: "success", icon: CircleCheck, spin: false, solid: false },
  FAILED: { label: "Failed", tone: "danger", icon: CircleX, spin: false, solid: false },
};

const forecast: Record<string, StatusMeta> = {
  QUEUED: { label: "Queued", tone: "neutral", icon: Clock, spin: false, solid: false },
  RUNNING: { label: "Running", tone: "info", icon: LoaderCircle, spin: true, solid: false },
  SUCCEEDED: { label: "Ready", tone: "success", icon: CircleCheck, spin: false, solid: false },
  FAILED: { label: "Failed", tone: "danger", icon: CircleX, spin: false, solid: false },
};

const dataset: Record<string, StatusMeta> = {
  registered: { label: "Not validated", tone: "neutral", icon: Circle, spin: false, solid: false },
  valid: { label: "Valid", tone: "success", icon: CircleCheck, spin: false, solid: false },
  invalid: { label: "Invalid", tone: "danger", icon: CircleX, spin: false, solid: false },
};

const model: Record<string, StatusMeta> = {
  TRAINED: { label: "Trained", tone: "neutral", icon: Circle, spin: false, solid: false },
  EVALUATED: { label: "Evaluated", tone: "neutral", icon: ChartColumn, spin: false, solid: false },
  PENDING_APPROVAL: {
    label: "Needs review",
    tone: "warning",
    icon: UserRoundCheck,
    spin: false,
    solid: false,
  },
  APPROVED: { label: "Approved", tone: "success", icon: Check, spin: false, solid: false },
  PRODUCTION: { label: "Production", tone: "success", icon: Star, spin: false, solid: true },
  REJECTED: { label: "Rejected", tone: "danger", icon: X, spin: false, solid: false },
};

export function statusMeta(
  kind: "training" | "forecast" | "dataset" | "model",
  value: string,
  reason?: string | null,
): StatusMeta {
  const table = { training, forecast, dataset, model }[kind];
  const found = table[value] ?? {
    label: value,
    tone: "neutral" as const,
    icon: Circle,
    spin: false,
    solid: false,
  };
  if (kind === "model" && value === "REJECTED") {
    if (reason === "quality_gate") {
      return { ...found, label: "Rejected · quality gate" };
    }
    if (reason === "human") {
      return { ...found, label: "Rejected · by reviewer" };
    }
  }
  return found;
}

export function familyLabel(family: string): string {
  switch (family) {
    case "naive":
      return "Naive";
    case "seasonal_naive":
      return "Seasonal naive";
    case "holt_winters":
      return "Holt-Winters";
    case "gradient_boosting":
      return "Gradient boosting";
    case "deepar":
      return "DeepAR";
    default:
      return family;
  }
}

export function familyToken(family: string): string {
  switch (family) {
    case "naive":
      return "--viz-naive";
    case "seasonal_naive":
      return "--viz-seasonal-naive";
    case "holt_winters":
      return "--viz-holt-winters";
    case "gradient_boosting":
      return "--viz-gradient-boosting";
    case "deepar":
      return "--viz-deepar";
    default:
      return "--text-muted";
  }
}

export function readToken(name: string): string {
  if (typeof document === "undefined") {
    return `var(${name})`;
  }
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || `var(${name})`;
}

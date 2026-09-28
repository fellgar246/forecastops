import { statusMeta } from "@/lib/status";

type StatusBadgeProps = {
  kind: "training" | "forecast" | "dataset" | "model";
  value: string;
  reason?: string | null;
  size?: "sm" | "md";
};

export function StatusBadge({ kind, value, reason, size = "sm" }: StatusBadgeProps) {
  const meta = statusMeta(kind, value, reason);
  const Icon = meta.icon;
  return (
    <span
      className={`badge badge-${size} badge-${meta.tone}${meta.solid ? " badge-solid" : ""}`}
      title={value}
    >
      <Icon aria-hidden="true" className={meta.spin ? "icon-spin" : undefined} size={size === "md" ? 16 : 14} />
      <span>{meta.label}</span>
    </span>
  );
}

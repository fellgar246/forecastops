export const MISSING_MARK = "—";

const numberFormat = new Intl.NumberFormat("en-US");
const currencyFormat = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function isMissing(value: number | null | undefined): value is null | undefined {
  return value == null || Number.isNaN(value);
}

export function formatCount(value: number | null | undefined): string {
  if (isMissing(value)) {
    return MISSING_MARK;
  }
  return numberFormat.format(Math.round(value));
}

export function formatUnits(value: number | null | undefined): string {
  if (isMissing(value)) {
    return MISSING_MARK;
  }
  return `${numberFormat.format(Math.round(value))} units`;
}

export function formatUnitsCompact(value: number | null | undefined): string {
  if (isMissing(value)) {
    return MISSING_MARK;
  }
  if (Math.abs(value) < 100_000) {
    return formatUnits(value);
  }
  return `${new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value)} units`;
}

export function formatRatioPercent(
  value: number | null | undefined,
  digits: number,
): string {
  if (isMissing(value)) {
    return MISSING_MARK;
  }
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatBias(value: number | null | undefined): string {
  if (isMissing(value)) {
    return MISSING_MARK;
  }
  const percent = Math.abs(value * 100).toFixed(1);
  const sign = value < 0 ? "−" : "+";
  return `${sign}${percent}%`;
}

export function formatDecimal(value: number | null | undefined, digits: number): string {
  if (isMissing(value)) {
    return MISSING_MARK;
  }
  return value.toFixed(digits);
}

export function formatMoney(value: number | null | undefined, estimated = false): string {
  if (isMissing(value)) {
    return MISSING_MARK;
  }
  const amount = currencyFormat.format(value);
  return estimated ? `≈ ${amount} (estimate)` : amount;
}

export function formatTokens(value: number | null | undefined): string {
  if (isMissing(value)) {
    return MISSING_MARK;
  }
  return `${numberFormat.format(Math.round(value))} tokens`;
}

export function formatDate(value: Date | string | null | undefined): string {
  if (value == null || value === "") {
    return MISSING_MARK;
  }
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) {
    return MISSING_MARK;
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

export function formatDuration(totalSeconds: number | null | undefined): string {
  if (isMissing(totalSeconds)) {
    return MISSING_MARK;
  }
  const seconds = Math.max(0, Math.round(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  const parts: string[] = [];
  if (hours > 0) {
    parts.push(`${hours}h`);
  }
  if (minutes > 0) {
    parts.push(`${minutes}m`);
  }
  if (rest > 0 || parts.length === 0) {
    parts.push(`${rest}s`);
  }
  return parts.slice(0, 2).join(" ");
}

export function formatHorizon(horizon: number, granularity: "day" | "week"): string {
  const unit = granularity === "week" ? "week" : "day";
  return `${numberFormat.format(horizon)} ${horizon === 1 ? unit : `${unit}s`}`;
}

export function formatRelative(
  value: Date | string | null | undefined,
  now: Date = new Date(),
): string {
  if (value == null || value === "") {
    return MISSING_MARK;
  }
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) {
    return MISSING_MARK;
  }
  const minutes = Math.round((now.getTime() - date.getTime()) / 60000);
  const elapsed = Math.abs(minutes);
  if (elapsed < 1) {
    return "just now";
  }
  if (elapsed < 60) {
    return elapsed === 1 ? "1 minute ago" : `${elapsed} minutes ago`;
  }
  const hours = Math.round(elapsed / 60);
  if (hours < 24) {
    return hours === 1 ? "1 hour ago" : `${hours} hours ago`;
  }
  const days = Math.round(hours / 24);
  return days === 1 ? "1 day ago" : `${days} days ago`;
}

export function formatId(value: string | null | undefined): string {
  if (value == null || value === "") {
    return MISSING_MARK;
  }
  if (value.length <= 12) {
    return value;
  }
  return `${value.slice(0, 4)}…${value.slice(-4)}`;
}

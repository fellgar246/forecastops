import { MISSING_MARK } from "@/lib/format";

export function MissingValue({ reason }: { reason: string }) {
  return (
    <span className="missing" title={reason}>
      {MISSING_MARK}
      <span className="visually-hidden"> {reason}</span>
    </span>
  );
}

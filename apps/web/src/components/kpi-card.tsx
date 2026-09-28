import type { ReactNode } from "react";
import { InfoTip } from "@/components/info-tip";
import type { GlossaryTerm } from "@/lib/glossary";

export function KpiCard({
  label,
  term,
  value,
  caption,
}: {
  label: string;
  term?: GlossaryTerm;
  value: ReactNode;
  caption: ReactNode;
}) {
  return (
    <article className="kpi-card">
      <p className="overline kpi-label">
        {label}
        {term ? <InfoTip term={term} /> : null}
      </p>
      <p className="kpi-value">{value}</p>
      <p className="kpi-caption">{caption}</p>
    </article>
  );
}

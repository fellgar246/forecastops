import { Info } from "lucide-react";
import { GLOSSARY, type GlossaryTerm } from "@/lib/glossary";

export function InfoTip({ term }: { term: GlossaryTerm }) {
  return (
    <span className="info-tip">
      <button type="button" className="info-tip-button" aria-label={term}>
        <Info aria-hidden="true" size={14} />
      </button>
      <span role="tooltip" className="info-tip-bubble">
        {GLOSSARY[term]}
      </span>
    </span>
  );
}

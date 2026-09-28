import { Check, Minus, X } from "lucide-react";
import type { ModelVersion } from "@/lib/api/schemas";
import { formatBias, formatRatioPercent, MISSING_MARK } from "@/lib/format";
import { familyLabel } from "@/lib/status";

export function QualityGatePanel({ model }: { model: ModelVersion }) {
  const promotion = model.promotion;
  if (!promotion) {
    return (
      <section className="card">
        <h2>Quality gate</h2>
        <p>No quality-gate record is stored for this model.</p>
      </section>
    );
  }
  const wape = model.metrics?.wape;
  const bias = model.metrics?.bias;
  const coverage = promotion.p90_coverage ?? model.metrics?.p90_coverage ?? null;
  const biasLimit = promotion.thresholds.max_bias;
  const coverageLimit = promotion.thresholds.min_p90_coverage;
  return (
    <section className="card" aria-labelledby="gate-title">
      <h2 id="gate-title">Quality gate</h2>
      <p className="caption">
        Compared with: {familyLabel(promotion.reference_id)} · dataset {model.dataset_version}
      </p>
      <ul className="gate-list">
        <GateRow
          state={promotion.checks.wape_improved ? "pass" : "fail"}
          rule="WAPE lower than the reference"
          value={wape == null ? MISSING_MARK : formatRatioPercent(wape, 1)}
        />
        <GateRow
          state={promotion.checks.bias_within_limit ? "pass" : "fail"}
          rule={
            biasLimit == null
              ? "Absolute bias within the stored limit"
              : `Absolute bias below ${formatRatioPercent(biasLimit, 0)}`
          }
          value={formatBias(bias)}
        />
        <GateRow
          state={
            promotion.checks.coverage_within_limit == null
              ? "na"
              : promotion.checks.coverage_within_limit
                ? "pass"
                : "fail"
          }
          rule={
            coverageLimit == null
              ? "P90 coverage meets the stored limit"
              : `P90 coverage at least ${formatRatioPercent(coverageLimit, 0)}`
          }
          value={
            promotion.checks.coverage_within_limit == null
              ? "Not applicable: model has no quantiles"
              : formatRatioPercent(coverage, 0)
          }
        />
        <GateRow
          state={promotion.checks.no_critical_segment_regression ? "pass" : "fail"}
          rule="No category regressed past the stored limit"
          value={
            promotion.regressed_categories.length === 0
              ? "None"
              : promotion.regressed_categories
                  .map(
                    (item) =>
                      `${item.category_id} ${formatRatioPercent(item.candidate_wape, 1)} vs ${formatRatioPercent(item.reference_wape, 1)}`,
                  )
                  .join("; ")
          }
        />
      </ul>
    </section>
  );
}

function GateRow({
  state,
  rule,
  value,
}: {
  state: "pass" | "fail" | "na";
  rule: string;
  value: string;
}) {
  const Icon = state === "pass" ? Check : state === "fail" ? X : Minus;
  return (
    <li className={`gate-row gate-${state}`}>
      <Icon aria-hidden="true" size={16} />
      <span>{rule}</span>
      <span className="numeric">{value}</span>
    </li>
  );
}

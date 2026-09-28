export const GLOSSARY = {
  WAPE: "Weighted absolute percentage error: total absolute error divided by total actual demand. Lower is better.",
  Bias: "Whether forecasts run high (positive) or low (negative) on average. Closer to zero is better.",
  P50: "The middle estimate: actual demand is expected to fall above or below it about equally often.",
  "P10–P90":
    "The likely range: actual demand is expected to fall inside it about 80% of the time.",
  "P90 coverage": "How often actual demand fell at or below P90 in backtests.",
  Horizon: "How far ahead the forecast goes.",
  Cutoff: "The last date of data the model could use.",
  "Seasonal naive": "A baseline that repeats last season's demand. New models must beat it.",
} as const;

export type GlossaryTerm = keyof typeof GLOSSARY;

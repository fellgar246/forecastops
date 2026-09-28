import type { ChartPoint } from "@/components/forecast-chart";
import type { ForecastSeries } from "@/lib/api/schemas";

export function chartPoints(series: ForecastSeries): ChartPoint[] {
  const byDate = new Map<string, ChartPoint>();
  for (const point of series.history) {
    byDate.set(point.date, {
      date: point.date,
      history: point.actual,
      p10: null,
      p50: null,
      p90: null,
      actual: null,
    });
  }
  for (const point of series.daily) {
    const existing = byDate.get(point.date);
    byDate.set(point.date, {
      date: point.date,
      history: existing?.history ?? null,
      p10: point.p10,
      p50: point.p50,
      p90: point.p90,
      actual: point.actual,
    });
  }
  return [...byDate.values()].sort((left, right) => left.date.localeCompare(right.date));
}

"use client";

import { useId, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatDate, formatUnits, MISSING_MARK } from "@/lib/format";
import { readToken } from "@/lib/status";

export type ChartPoint = {
  date: string;
  history: number | null;
  p10: number | null;
  p50: number | null;
  p90: number | null;
  actual: number | null;
};

type Row = ChartPoint & { band: [number, number] | null };

export function ForecastChart({
  points,
  cutoff,
  subtitle,
  modelVersion,
}: {
  points: ChartPoint[];
  cutoff: string | null;
  subtitle: string;
  modelVersion: string | null;
}) {
  const summaryId = useId();
  const [table, setTable] = useState(false);
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const rows: Row[] = points.map((point) => ({
    ...point,
    band: point.p10 != null && point.p90 != null ? [point.p10, point.p90] : null,
  }));
  const hasQuantiles = rows.some((point) => point.p10 != null || point.p90 != null);
  const summary = chartSummary(points);
  const colors = {
    history: readToken("--viz-history"),
    p50: readToken("--viz-p50"),
    band: readToken("--viz-band"),
    actual: readToken("--viz-actual"),
    cutoff: readToken("--viz-cutoff"),
    grid: readToken("--border"),
    text: readToken("--text-muted"),
  };
  return (
    <section className="card chart-card" aria-labelledby={summaryId}>
      <div className="card-header">
        <div>
          <h2>Demand</h2>
          <p className="caption">{subtitle}</p>
        </div>
        <button type="button" className="btn btn-secondary" onClick={() => setTable((value) => !value)}>
          {table ? "View as chart" : "View as table"}
        </button>
      </div>
      <p id={summaryId} className="chart-summary">
        {summary}
      </p>
      {table ? (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th className="numeric">History</th>
                <th className="numeric">Actual</th>
                <th className="numeric">P90</th>
                <th className="numeric">P50</th>
                <th className="numeric">P10</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((point) => (
                <tr key={point.date}>
                  <td>{formatDate(point.date)}</td>
                  <td className="numeric">{formatUnits(point.history)}</td>
                  <td className="numeric">{formatUnits(point.actual)}</td>
                  <td className="numeric">{formatUnits(point.p90)}</td>
                  <td className="numeric">{formatUnits(point.p50)}</td>
                  <td className="numeric">{formatUnits(point.p10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="chart-frame" role="img" aria-label="Demand history and forecast" aria-describedby={summaryId}>
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={rows}>
              <CartesianGrid stroke={colors.grid} vertical={false} />
              <XAxis dataKey="date" tickFormatter={(value: string) => formatDate(value)} tick={{ fill: colors.text, fontSize: 12 }} />
              <YAxis domain={[0, "auto"]} tick={{ fill: colors.text, fontSize: 12 }} />
              <Tooltip content={<ForecastTooltip modelVersion={modelVersion} />} />
              <Legend
                onClick={(entry) => {
                  const key = String(entry.dataKey ?? "");
                  if (!key) {
                    return;
                  }
                  setHidden((current) => ({ ...current, [key]: !current[key] }));
                }}
              />
              {cutoff ? (
                <ReferenceLine x={cutoff} stroke={colors.cutoff} strokeDasharray="3 3" label="Forecast starts" />
              ) : null}
              {!hidden.band ? (
                <Area dataKey="band" name="P10–P90" stroke="none" fill={colors.band} legendType="rect" />
              ) : null}
              {!hidden.history ? (
                <Line dataKey="history" name="History" stroke={colors.history} dot={false} strokeWidth={2} connectNulls />
              ) : null}
              {!hidden.p50 ? (
                <Line dataKey="p50" name="P50" stroke={colors.p50} dot={false} strokeWidth={2} connectNulls />
              ) : null}
              {!hidden.actual ? (
                <Scatter dataKey="actual" name="Actual" fill={colors.actual} />
              ) : null}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
      {hasQuantiles ? null : (
        <p className="caption">This model returns a single value. No uncertainty range is available.</p>
      )}
    </section>
  );
}

function chartSummary(points: ChartPoint[]): string {
  const forecast = points.filter((point) => point.p50 != null);
  const first = forecast[0];
  const last = forecast[forecast.length - 1];
  if (!first || !last || first.p50 == null || last.p50 == null) {
    return "No forecast points are available for this scope.";
  }
  const direction = last.p50 > first.p50 ? "rises" : last.p50 < first.p50 ? "falls" : "stays level";
  return `P50 ${direction} from ${formatUnits(first.p50)} to ${formatUnits(last.p50)} over ${forecast.length} points.`;
}

function ForecastTooltip({
  active,
  payload,
  label,
  modelVersion,
}: {
  active?: boolean;
  payload?: { payload?: Row }[];
  label?: string;
  modelVersion: string | null;
}) {
  if (!active || !payload?.length) {
    return null;
  }
  const point = payload[0]?.payload;
  if (!point) {
    return null;
  }
  return (
    <div className="chart-tooltip">
      <p>{formatDate(label ?? point.date)}</p>
      <p>Actual {formatUnits(point.actual)}</p>
      <p>P90 {point.p90 == null ? MISSING_MARK : formatUnits(point.p90)}</p>
      <p>P50 {formatUnits(point.p50)}</p>
      <p>P10 {point.p10 == null ? MISSING_MARK : formatUnits(point.p10)}</p>
      <p className="mono">{modelVersion ?? MISSING_MARK}</p>
    </div>
  );
}

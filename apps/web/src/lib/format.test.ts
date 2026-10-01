import { expect, test } from "vitest";
import {
  formatBias,
  formatDate,
  formatDateTime,
  formatDuration,
  formatMoney,
  formatRatioPercent,
  formatUnits,
  MISSING_MARK,
} from "./format";

test("formats demand and missing values", () => {
  expect(formatUnits(1840)).toBe("1,840 units");
  expect(formatUnits(null)).toBe(MISSING_MARK);
});

test("formats ratios with a true minus sign for negative bias", () => {
  expect(formatRatioPercent(0.132, 1)).toBe("13.2%");
  expect(formatBias(0.004)).toBe("+0.4%");
  expect(formatBias(-0.021)).toBe("−2.1%");
  expect(formatBias(null)).toBe(MISSING_MARK);
});

test("formats money, dates, and durations", () => {
  expect(formatMoney(3.4, true)).toBe("≈ $3.40 (estimate)");
  expect(formatDate("2026-09-21T00:00:00Z")).toBe("Sep 21, 2026");
  expect(formatDateTime("2026-09-30T16:05:00Z")).toBe("Sep 30, 2026, 4:05 PM UTC");
  expect(formatDuration(453)).toBe("7m 33s");
  expect(formatDate(null)).toBe(MISSING_MARK);
});

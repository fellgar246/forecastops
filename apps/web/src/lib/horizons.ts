export type HorizonChoice = {
  horizon: number;
  granularity: "day" | "week";
  label: string;
};

export function horizonChoices(maxDays: number): HorizonChoice[] {
  const days = [7, 14, 28, 56, 90].filter((day) => day <= maxDays);
  if (maxDays > 0 && !days.includes(maxDays)) {
    days.push(maxDays);
  }
  const weeks = [4, 8, 12].filter((week) => week * 7 <= maxDays);
  return [
    ...days.map((horizon) => ({
      horizon,
      granularity: "day" as const,
      label: `${horizon} days`,
    })),
    ...weeks.map((horizon) => ({
      horizon,
      granularity: "week" as const,
      label: `${horizon} weeks`,
    })),
  ];
}

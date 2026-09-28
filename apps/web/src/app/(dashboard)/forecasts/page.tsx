import type { Metadata } from "next";
import { Suspense } from "react";
import { ForecastsPage } from "@/features/forecasts/forecasts-page";

export const metadata: Metadata = { title: "Forecasts · ForecastOps" };

export default function Page() {
  return (
    <Suspense fallback={<p className="page">Loading filters…</p>}>
      <ForecastsPage />
    </Suspense>
  );
}

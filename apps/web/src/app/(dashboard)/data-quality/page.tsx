import type { Metadata } from "next";
import { Suspense } from "react";
import { DataQualityPage } from "@/features/quality/data-quality-page";

export const metadata: Metadata = { title: "Data quality · ForecastOps" };

export default function Page() {
  return (
    <Suspense fallback={<p className="page">Loading data quality…</p>}>
      <DataQualityPage />
    </Suspense>
  );
}

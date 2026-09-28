import type { Metadata } from "next";
import { Suspense } from "react";
import { ComparePage } from "@/features/models/compare-page";

export const metadata: Metadata = { title: "Compare · ForecastOps" };

export default function Page() {
  return (
    <Suspense fallback={<p className="page">Loading comparison…</p>}>
      <ComparePage />
    </Suspense>
  );
}

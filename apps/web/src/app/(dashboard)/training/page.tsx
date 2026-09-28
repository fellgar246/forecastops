import type { Metadata } from "next";
import { Suspense } from "react";
import { TrainingPage } from "@/features/training/training-page";

export const metadata: Metadata = { title: "Training · ForecastOps" };

export default function Page() {
  return (
    <Suspense fallback={<p className="page">Loading training…</p>}>
      <TrainingPage />
    </Suspense>
  );
}

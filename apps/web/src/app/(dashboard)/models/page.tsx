import type { Metadata } from "next";
import { Suspense } from "react";
import { ModelsPage } from "@/features/models/models-page";

export const metadata: Metadata = { title: "Models · ForecastOps" };

export default function Page() {
  return (
    <Suspense fallback={<p className="page">Loading models…</p>}>
      <ModelsPage />
    </Suspense>
  );
}

import type { Metadata } from "next";
import { ForecastDetailPage } from "@/features/forecasts/forecast-detail-page";

export const metadata: Metadata = { title: "Forecast · ForecastOps" };

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <ForecastDetailPage id={id} />;
}

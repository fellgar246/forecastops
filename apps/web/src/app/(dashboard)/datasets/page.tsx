import type { Metadata } from "next";
import { DatasetsPage } from "@/features/datasets/datasets-page";

export const metadata: Metadata = { title: "Datasets · ForecastOps" };

export default function Page() {
  return <DatasetsPage />;
}

import type { Metadata } from "next";
import { CostPage } from "@/features/cost/cost-page";

export const metadata: Metadata = { title: "Cost · ForecastOps" };

export default function Page() {
  return <CostPage />;
}
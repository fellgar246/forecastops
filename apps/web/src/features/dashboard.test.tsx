import { fireEvent, screen } from "@testing-library/react";
import { beforeEach, expect, test } from "vitest";
import { ExplanationPanel } from "@/components/explanation-panel";
import { ForecastChart } from "@/components/forecast-chart";
import { CostPage } from "@/features/cost/cost-page";
import { DatasetsPage } from "@/features/datasets/datasets-page";
import { ForecastsPage } from "@/features/forecasts/forecasts-page";
import { ComparePage } from "@/features/models/compare-page";
import { ModelsPage } from "@/features/models/models-page";
import { OverviewPage } from "@/features/overview/overview-page";
import { DataQualityPage } from "@/features/quality/data-quality-page";
import { TrainingPage } from "@/features/training/training-page";
import { jsonResponse, mockFetch, renderPage } from "@/test/render";

const health = {
  status: "healthy",
  execution_mode: "local",
  aws_enabled: false,
  bedrock_enabled: false,
  sagemaker_enabled: false,
  training_enabled: true,
  max_forecast_horizon_days: 90,
  max_training_jobs_per_day: 2,
};

const aws = {
  aws_enabled: false,
  aws_ml_enabled: false,
  bedrock_enabled: false,
  sagemaker_enabled: false,
  online_inference: false,
};

const metrics = {
  row_count: 8,
  mae: 1.2,
  rmse: 1.4,
  wape: 0.132,
  smape: 0.11,
  bias: 0.004,
  pinball_loss_p10: null,
  pinball_loss_p90: null,
  p90_coverage: null,
};

function pendingModel(status = "PENDING_APPROVAL") {
  return {
    id: "model-1",
    model_family: "seasonal_naive",
    version: "1",
    training_run_id: "run-1",
    dataset_id: "dataset-1",
    dataset_version: "1",
    registry_arn: "",
    registry_status: "",
    status,
    metrics,
    rejection_reason: null,
    promotion: {
      reference_id: "seasonal_naive",
      thresholds: { max_bias: 0.05, min_p90_coverage: 0.85, max_segment_wape_regression: 0.02 },
      checks: {
        wape_improved: true,
        bias_within_limit: true,
        coverage_within_limit: null,
        no_critical_segment_regression: true,
      },
      reason: null,
      p90_coverage: null,
      regressed_categories: [],
    },
    approved_at: null,
    created_at: "2026-09-21T00:00:00Z",
  };
}

beforeEach(() => {
  const nav = (globalThis as unknown as { __forecastopsNav: { pathname: string; search: string } })
    .__forecastopsNav;
  nav.pathname = "/overview";
  nav.search = "";
});

test("pages explain what is missing when nothing is registered", async () => {
  mockFetch((url) => {
    if (url.pathname === "/health") return jsonResponse(health);
    if (url.pathname === "/health/aws") return jsonResponse(aws);
    return jsonResponse({ items: [] });
  });

  const pages = [
    { ui: <OverviewPage />, name: "No forecasts yet" },
    { ui: <ForecastsPage />, name: "No forecasts to show" },
    { ui: <ModelsPage />, name: "No models yet" },
    { ui: <ComparePage />, name: "Nothing to compare yet" },
    { ui: <TrainingPage />, name: "No training runs yet" },
    { ui: <DatasetsPage />, name: "No datasets registered" },
    { ui: <DataQualityPage />, name: "No quality report yet" },
  ];
  for (const page of pages) {
    const view = renderPage(page.ui);
    expect(await screen.findByRole("heading", { name: page.name })).toBeVisible();
    view.unmount();
  }
  const cost = renderPage(<CostPage />);
  expect(await screen.findByText("Cloud spend tracking is inactive.")).toBeVisible();
  cost.unmount();
});

test("overview shows the production snapshot from the API", async () => {
  const model = { ...pendingModel("PRODUCTION"), approved_at: "2026-09-21T00:00:00Z" };
  mockFetch((url) => {
    if (url.pathname === "/health") return jsonResponse(health);
    if (url.pathname === "/models" || url.pathname === "/metrics/model-performance") {
      return jsonResponse({ items: [model] });
    }
    if (url.pathname === "/training-runs") {
      return jsonResponse({
        items: [
          {
            id: "run-1",
            dataset_id: "dataset-1",
            dataset_version: "1",
            model_family: "seasonal_naive",
            configuration: {},
            status: "COMPLETED",
            started_at: "2026-09-21T00:00:00Z",
            finished_at: "2026-09-21T00:05:00Z",
            artifact_uri: "",
            metrics: null,
            error_message: null,
            git_sha: "",
            pipeline_execution_arn: "",
            created_at: "2026-09-21T00:00:00Z",
          },
        ],
      });
    }
    if (url.pathname === "/forecasts") {
      return jsonResponse({
        items: [
          {
            id: "forecast-1",
            model_id: "model-1",
            model_family: "seasonal_naive",
            model_version: "1",
            dataset_id: "dataset-1",
            dataset_version: "1",
            horizon: 7,
            granularity: "day",
            status: "SUCCEEDED",
            output_uri: "",
            error_message: null,
            created_at: "2026-09-21T01:00:00Z",
          },
        ],
      });
    }
    if (url.pathname === "/forecasts/forecast-1/series") {
      return jsonResponse({
        items: [
          { series_id: "store-1|sku-1", date: "2026-10-01", p10: 8, p50: 10, p90: 12, actual: null },
        ],
        series: [
          { series_id: "store-1|sku-1", point_count: 1, p10_total: 8, p50_total: 10, p90_total: 12 },
        ],
        history: [{ date: "2026-09-30", actual: 9 }],
        daily: [{ date: "2026-10-01", p10: 8, p50: 10, p90: 12, actual: null }],
        cutoff: "2026-10-01",
        p10_total: 8,
        p50_total: 10,
        p90_total: 12,
      });
    }
    return jsonResponse({ items: [] });
  });

  renderPage(<OverviewPage />);
  expect((await screen.findAllByText("13.2%")).length).toBeGreaterThan(0);
  expect(screen.getAllByText("10 units").length).toBeGreaterThan(0);
  expect(screen.getAllByText("Seasonal naive 1").length).toBeGreaterThan(0);
  expect(screen.getAllByText("7 days").length).toBeGreaterThan(0);
  expect(screen.getByRole("img", { name: "Demand history and forecast" })).toBeVisible();
});

test("approving a pending model calls the API and shows the new status", async () => {
  let status = "PENDING_APPROVAL";
  const fetchMock = mockFetch((url, init) => {
    if (url.pathname === "/models/model-1/approve" && init?.method === "POST") {
      status = "APPROVED";
      return jsonResponse(pendingModel("APPROVED"));
    }
    if (url.pathname === "/models") {
      return jsonResponse({ items: [pendingModel(status)] });
    }
    if (url.pathname === "/health") return jsonResponse(health);
    return jsonResponse({ items: [] });
  });

  renderPage(<ModelsPage />);
  fireEvent.click(await screen.findByRole("button", { name: "Review" }));
  fireEvent.click(screen.getByRole("button", { name: "Approve" }));
  fireEvent.click(screen.getByRole("button", { name: "Approve model" }));

  expect((await screen.findAllByText("Approved")).length).toBeGreaterThan(0);
  const approveCall = fetchMock.mock.calls.find((call) => String(call[0]).includes("/approve"));
  expect(approveCall?.[1]).toMatchObject({ method: "POST" });
  expect(JSON.parse(String(approveCall?.[1]?.body))).toEqual({
    actor_id: "local-reviewer",
    promote: false,
  });
});

test("forecast filters are sent to the series endpoint", async () => {
  (globalThis as unknown as { __forecastopsNav: { search: string } }).__forecastopsNav.search =
    "category=grocery&store=store-1&horizon=7";
  const fetchMock = mockFetch((url) => {
    if (url.pathname === "/health") return jsonResponse(health);
    if (url.pathname === "/forecasts") {
      return jsonResponse({
        items: [
          {
            id: "forecast-1",
            model_id: "model-1",
            model_family: "seasonal_naive",
            model_version: "1",
            dataset_id: "dataset-1",
            dataset_version: "1",
            horizon: 7,
            granularity: "day",
            status: "SUCCEEDED",
            output_uri: "",
            error_message: null,
            created_at: "2026-09-21T01:00:00Z",
          },
        ],
      });
    }
    if (url.pathname === "/datasets/dataset-1/catalog") {
      return jsonResponse({
        stores: [{ id: "store-1", name: "store-1" }],
        categories: [{ id: "grocery", name: "Grocery" }],
        skus: [{ id: "sku-1", category_id: "grocery" }],
      });
    }
    if (url.pathname === "/forecasts/forecast-1/series") {
      return jsonResponse({
        items: [],
        series: [],
        history: [],
        daily: [],
        cutoff: null,
        p10_total: null,
        p50_total: null,
        p90_total: null,
      });
    }
    return jsonResponse({ items: [] });
  });

  renderPage(<ForecastsPage />);
  expect(await screen.findByRole("heading", { name: "No series match these filters" })).toBeVisible();
  const seriesCall = fetchMock.mock.calls.map((call) => String(call[0])).find((url) => url.includes("/series"));
  expect(seriesCall).toContain("category=grocery");
  expect(seriesCall).toContain("store=store-1");
  expect(seriesCall).toContain("horizon=7");
});

test("explanation panel states that nothing has been generated on 409", () => {
  renderPage(
    <ExplanationPanel view={{ kind: "disabled", message: "Explanations are not enabled." }} />,
  );
  expect(
    screen.getByText("No explanation has been generated yet. Explanations are turned off in this environment."),
  ).toBeVisible();
});

test("forecast chart says when quantiles are missing", () => {
  renderPage(
    <ForecastChart
      points={[{ date: "2026-10-01", history: null, p10: null, p50: 10, p90: null, actual: null }]}
      cutoff="2026-10-01"
      subtitle="Daily units · 7-day horizon"
      modelVersion="1"
    />,
  );
  expect(screen.getByText("This model returns a single value. No uncertainty range is available.")).toBeVisible();
});

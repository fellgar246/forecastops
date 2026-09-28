import { expect, test } from "@playwright/test";

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

test("overview loads against a mocked local API", async ({ page }) => {
  await page.route(/https?:\/\/(localhost:8000|127\.0\.0\.1:8010)\//, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/health") {
      await route.fulfill({ json: health });
      return;
    }
    if (path === "/health/aws") {
      await route.fulfill({ json: aws });
      return;
    }
    await route.fulfill({ json: { items: [] } });
  });

  await page.goto("/overview");
  await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "No forecasts yet" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Environment Local" })).toBeVisible();
});

import { cleanup, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { AppShell } from "@/components/app-shell";
import { jsonResponse, mockFetch, renderPage } from "@/test/render";

afterEach(() => {
  cleanup();
  vi.unstubAllEnvs();
  window.sessionStorage.clear();
});

test("local mode hides the sign-in form", () => {
  vi.stubEnv("NEXT_PUBLIC_AUTH_ENABLED", "false");
  mockFetch((url) => {
    if (url.pathname === "/health") {
      return jsonResponse({
        status: "healthy",
        execution_mode: "local",
        aws_enabled: false,
        bedrock_enabled: false,
        sagemaker_enabled: false,
        training_enabled: true,
        max_forecast_horizon_days: 90,
        max_training_jobs_per_day: 2,
      });
    }
    if (url.pathname === "/health/aws") {
      return jsonResponse({
        aws_enabled: false,
        aws_ml_enabled: false,
        bedrock_enabled: false,
        sagemaker_enabled: false,
        online_inference: false,
      });
    }
    return jsonResponse({ items: [] });
  });

  renderPage(
    <AppShell>
      <h1>Overview</h1>
    </AppShell>,
  );

  expect(screen.getByRole("heading", { name: "Overview" })).toBeVisible();
  expect(screen.queryByRole("heading", { name: "Sign in" })).not.toBeInTheDocument();
});

test("cloud auth shows the sign-in form until a token is stored", () => {
  vi.stubEnv("NEXT_PUBLIC_AUTH_ENABLED", "true");

  renderPage(
    <AppShell>
      <h1>Overview</h1>
    </AppShell>,
  );

  expect(screen.getByRole("heading", { name: "Sign in" })).toBeVisible();
  expect(screen.queryByRole("heading", { name: "Overview" })).not.toBeInTheDocument();
});

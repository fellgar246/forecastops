import { test, expect } from "@playwright/test";

test("home page describes the local profile", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Demand forecasts for retail" })).toBeVisible();
  await expect(page.getByText("Environment")).toBeVisible();
  await expect(page.getByText("Local", { exact: true })).toBeVisible();
});

import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import HomePage from "./page";

test("shows the local profile placeholder", () => {
  render(<HomePage />);

  expect(screen.getByRole("heading", { name: "Demand forecasts for retail" })).toBeVisible();
  expect(screen.getByText(/cloud training, hosted inference/i)).toBeVisible();
  expect(screen.getByText("Local")).toBeVisible();
  expect(screen.getByText("No dataset registered")).toBeVisible();
});

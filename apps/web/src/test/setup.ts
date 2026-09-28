import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

type NavState = {
  pathname: string;
  search: string;
  push: (href: string) => void;
  replace: (href: string) => void;
};

const navigation: NavState = {
  pathname: "/overview",
  search: "",
  push: vi.fn(),
  replace: vi.fn(),
};

vi.stubGlobal("__forecastopsNav", navigation);

vi.mock("next/navigation", () => ({
  usePathname: () =>
    (globalThis as typeof globalThis & { __forecastopsNav: NavState }).__forecastopsNav.pathname,
  useSearchParams: () =>
    new URLSearchParams(
      (globalThis as typeof globalThis & { __forecastopsNav: NavState }).__forecastopsNav.search,
    ),
  useRouter: () => ({
    push: (href: string) =>
      (globalThis as typeof globalThis & { __forecastopsNav: NavState }).__forecastopsNav.push(href),
    replace: (href: string) =>
      (globalThis as typeof globalThis & { __forecastopsNav: NavState }).__forecastopsNav.replace(href),
    refresh: () => undefined,
  }),
  redirect: (url: string) => {
    throw new Error(`redirect:${url}`);
  },
}));

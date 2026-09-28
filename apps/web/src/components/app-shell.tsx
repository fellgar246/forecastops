"use client";

import {
  Boxes,
  ChartLine,
  Database,
  GitCompare,
  LayoutDashboard,
  Menu,
  Play,
  ShieldCheck,
  Wallet,
  type LucideIcon,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";
import { EnvironmentChip } from "@/components/environment-chip";
import { useModels, useTrainingRuns } from "@/lib/api/queries";

type Item = {
  href: string;
  label: string;
  icon: LucideIcon;
  badge?: "models" | "training";
};

const GROUPS: { label?: string; items: Item[] }[] = [
  {
    items: [
      { href: "/overview", label: "Overview", icon: LayoutDashboard },
      { href: "/forecasts", label: "Forecasts", icon: ChartLine },
    ],
  },
  {
    label: "Models",
    items: [
      { href: "/models", label: "Models", icon: Boxes, badge: "models" },
      { href: "/models/compare", label: "Compare", icon: GitCompare },
      { href: "/training", label: "Training", icon: Play, badge: "training" },
    ],
  },
  {
    label: "Data",
    items: [
      { href: "/datasets", label: "Datasets", icon: Database },
      { href: "/data-quality", label: "Data quality", icon: ShieldCheck },
    ],
  },
  {
    label: "Platform",
    items: [{ href: "/cost", label: "Cost", icon: Wallet }],
  },
];

function active(pathname: string, href: string): boolean {
  if (href === "/models") {
    return pathname === "/models";
  }
  if (href === "/forecasts") {
    return pathname === "/forecasts" || pathname.startsWith("/forecasts/");
  }
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const models = useModels();
  const training = useTrainingRuns();
  const pending = models.data?.items.filter((model) => model.status === "PENDING_APPROVAL").length ?? 0;
  const running =
    training.data?.items.filter((run) => run.status !== "COMPLETED" && run.status !== "FAILED").length ?? 0;
  return (
    <div className="shell">
      <button type="button" className="menu-button" aria-label="Open navigation" onClick={() => setOpen(true)}>
        <Menu size={20} />
      </button>
      <aside className={open ? "sidebar sidebar-open" : "sidebar"}>
        <Link href="/overview" className="sidebar-brand">
          <span className="brand-mark" aria-hidden="true">
            F
          </span>
          <span className="brand-name">ForecastOps</span>
        </Link>
        <nav className="sidebar-nav" aria-label="Primary">
          {GROUPS.map((group) => (
            <div key={group.label ?? "top"}>
              {group.label ? <p className="nav-group">{group.label}</p> : null}
              <ul>
                {group.items.map((item) => {
                  const Icon = item.icon;
                  const count = item.badge === "models" ? pending : item.badge === "training" ? running : 0;
                  const current = active(pathname, item.href);
                  return (
                    <li key={item.href}>
                      <Link
                        href={item.href}
                        className="nav-link"
                        aria-current={current ? "page" : undefined}
                        onClick={() => setOpen(false)}
                      >
                        <Icon className="nav-icon" size={20} aria-hidden="true" />
                        <span className="nav-label">{item.label}</span>
                        {count > 0 ? <span className="nav-badge">{count}</span> : null}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <EnvironmentChip />
        </div>
      </aside>
      {open ? <button type="button" className="sidebar-scrim" aria-label="Close navigation" onClick={() => setOpen(false)} /> : null}
      <div className="content">{children}</div>
    </div>
  );
}

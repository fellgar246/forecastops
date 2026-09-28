import type { LucideIcon } from "lucide-react";
import { CircleDashed } from "lucide-react";
import Link from "next/link";

export function EmptyState({
  title,
  body,
  action,
  icon: Icon = CircleDashed,
}: {
  title: string;
  body: string;
  action?: { href: string; label: string };
  icon?: LucideIcon;
}) {
  return (
    <section className="empty-state">
      <Icon aria-hidden="true" size={48} />
      <h2>{title}</h2>
      <p>{body}</p>
      {action ? (
        <Link className="btn btn-primary" href={action.href}>
          {action.label}
        </Link>
      ) : null}
    </section>
  );
}

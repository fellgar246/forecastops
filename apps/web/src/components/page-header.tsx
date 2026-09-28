import type { ReactNode } from "react";

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <h1>{title}</h1>
        <p className="lede">{description}</p>
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </header>
  );
}

export function ContextBar({ items }: { items: { label: string; value: ReactNode }[] }) {
  return (
    <section className="context-bar" aria-label="Page context">
      {items.map((item) => (
        <p key={item.label}>
          <span className="context-label">{item.label}</span> {item.value}
        </p>
      ))}
    </section>
  );
}

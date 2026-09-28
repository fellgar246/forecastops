"use client";

import { useEffect, useState } from "react";
import type { DatasetCatalog } from "@/lib/api/schemas";

export type ForecastFilters = {
  category: string;
  sku: string;
  store: string;
  horizon: string;
};

export function FilterBar({
  catalog,
  horizons,
  values,
  onChange,
}: {
  catalog: DatasetCatalog | undefined;
  horizons: number[];
  values: ForecastFilters;
  onChange: (next: ForecastFilters) => void;
}) {
  const [sku, setSku] = useState(values.sku);
  const [store, setStore] = useState(values.store);
  useEffect(() => {
    setSku(values.sku);
    setStore(values.store);
  }, [values.sku, values.store]);
  useEffect(() => {
    const handle = window.setTimeout(() => {
      if (sku !== values.sku || store !== values.store) {
        onChange({ ...values, sku, store });
      }
    }, 300);
    return () => window.clearTimeout(handle);
  }, [sku, store, values, onChange]);
  const skus = (catalog?.skus ?? []).filter(
    (item) => values.category === "" || item.category_id === values.category,
  );
  const active = [
    values.category ? { key: "category" as const, label: values.category } : null,
    values.sku ? { key: "sku" as const, label: values.sku } : null,
    values.store ? { key: "store" as const, label: values.store } : null,
    values.horizon ? { key: "horizon" as const, label: `${values.horizon}` } : null,
  ].filter((item) => item != null);
  return (
    <div className="filter-bar">
      <label className="field">
        Category
        <select
          aria-label="Category"
          value={values.category}
          onChange={(event) => onChange({ ...values, category: event.target.value, sku: "" })}
        >
          <option value="">All categories</option>
          {(catalog?.categories ?? []).map((item) => (
            <option key={item.id} value={item.id}>
              {item.name}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        SKU
        <input
          aria-label="SKU"
          list="sku-options"
          value={sku}
          onChange={(event) => setSku(event.target.value)}
        />
        <datalist id="sku-options">
          {skus.map((item) => (
            <option key={item.id} value={item.id} />
          ))}
        </datalist>
      </label>
      <label className="field">
        Store
        <input
          aria-label="Store"
          list="store-options"
          value={store}
          onChange={(event) => setStore(event.target.value)}
        />
        <datalist id="store-options">
          {(catalog?.stores ?? []).map((item) => (
            <option key={item.id} value={item.id}>
              {item.name}
            </option>
          ))}
        </datalist>
      </label>
      <label className="field">
        Horizon
        <select
          aria-label="Horizon"
          value={values.horizon}
          onChange={(event) => onChange({ ...values, horizon: event.target.value })}
        >
          <option value="">All horizons</option>
          {horizons.map((horizon) => (
            <option key={horizon} value={String(horizon)}>
              {horizon}
            </option>
          ))}
        </select>
      </label>
      {active.length > 0 ? (
        <div className="chip-row">
          {active.map((item) => (
            <button
              key={item.key}
              type="button"
              className="chip"
              onClick={() => onChange({ ...values, [item.key]: "" })}
            >
              {item.label} <span aria-hidden="true">×</span>
              <span className="visually-hidden">Remove {item.label}</span>
            </button>
          ))}
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => onChange({ category: "", sku: "", store: "", horizon: "" })}
          >
            Clear all
          </button>
        </div>
      ) : null}
    </div>
  );
}

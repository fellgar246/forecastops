"""Deterministic synthetic retail demand history.

``make seed-data`` writes the full demo catalog. Tests use the ``test`` profile:
two stores, four SKUs, and eight weeks. :func:`build_plan` inspects a profile's
catalog and row count without simulating demand or writing files.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from numpy.typing import NDArray

from forecastops_contracts import (
    CATEGORY_COLUMNS,
    OBSERVATION_COLUMNS,
    SCHEMA_VERSION,
    SKU_COLUMNS,
    STORE_COLUMNS,
    ColumnSpec,
    LogicalType,
    category_schema_problems,
    observation_schema_problems,
    sku_schema_problems,
    store_schema_problems,
)

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
I32 = NDArray[np.int32]
BoolArray = NDArray[np.bool_]

DEFAULT_SEED = 20260921
DEFAULT_MAX_ROWS = 2_000_000
_UNIX_EPOCH = date(1970, 1, 1)
_MATURE = "mature"
_NEW = "new"
_DECLINING = "declining"
_REGIONS = ("north", "south", "central", "west")
_REGION_PHASE = (20.0, 200.0, 80.0, 140.0)
_BRANDS = ("Northwind", "Lumen", "Campo", "Harbor", "Alto", "Nube")
_CAMPAIGNS = ("weekend-lift", "category-push", "clearance", "holiday-event")
_DISCOUNTS = np.array([5.0, 10.0, 15.0, 20.0, 25.0], dtype=np.float64)
_HOLIDAYS = ((1, 1), (5, 1), (9, 16), (11, 1), (12, 24), (12, 25), (12, 31))

_ARROW: dict[LogicalType, pa.DataType] = {
    LogicalType.DATE: pa.date32(),
    LogicalType.STRING: pa.string(),
    LogicalType.INT64: pa.int64(),
    LogicalType.FLOAT64: pa.float64(),
    LogicalType.BOOLEAN: pa.bool_(),
}


class DatasetSummary(TypedDict):
    """Fields written to ``summary.json``."""

    categories: int
    date_max: str
    date_min: str
    profile: str
    row_count: int
    schema_version: str
    seed: int
    skus: int
    stores: int


@dataclass(frozen=True)
class _CategoryTemplate:
    category_id: str
    category_name: str
    base_cents: int
    base_demand: float
    weekly_amplitude: float
    annual_amplitude: float
    lead_time_days: int


_CATEGORIES = (
    _CategoryTemplate("beverages", "Beverages", 249, 16.0, 0.22, 0.12, 5),
    _CategoryTemplate("snacks", "Snacks", 199, 11.0, 0.18, 0.10, 7),
    _CategoryTemplate("dairy", "Dairy", 349, 14.0, 0.08, 0.06, 3),
    _CategoryTemplate("bakery", "Bakery", 279, 9.0, 0.28, 0.15, 2),
    _CategoryTemplate("household", "Household", 699, 4.0, 0.06, 0.08, 14),
    _CategoryTemplate("personal-care", "Personal Care", 849, 3.0, 0.05, 0.07, 21),
    _CategoryTemplate("frozen", "Frozen", 529, 7.0, 0.10, 0.14, 10),
    _CategoryTemplate("produce", "Produce", 179, 13.0, 0.12, 0.20, 2),
)


@dataclass(frozen=True)
class GenerationProfile:
    """Catalog size and date window for one generator run."""

    name: str
    n_stores: int
    n_skus: int
    n_categories: int
    date_min: date
    date_max: date
    stock_all: bool


_PROFILES: dict[str, GenerationProfile] = {
    "default": GenerationProfile(
        name="default",
        n_stores=20,
        n_skus=150,
        n_categories=8,
        date_min=date(2024, 10, 1),
        date_max=date(2026, 9, 30),
        stock_all=False,
    ),
    "test": GenerationProfile(
        name="test",
        n_stores=2,
        n_skus=4,
        n_categories=2,
        date_min=date(2026, 8, 6),
        date_max=date(2026, 9, 30),
        stock_all=True,
    ),
}


@dataclass(frozen=True)
class StoreRecord:
    """One store in the generated catalog."""

    index: int
    store_id: str
    store_region: str
    scale: float
    region_phase: float


@dataclass(frozen=True)
class SkuRecord:
    """One SKU in the generated catalog."""

    index: int
    sku_id: str
    category_id: str
    product_brand: str
    product_lifecycle: str
    list_price: float
    base_demand: float
    weekly_amplitude: float
    annual_amplitude: float
    lead_time_days: int


@dataclass(frozen=True)
class CategoryRecord:
    """One category in the generated catalog."""

    category_id: str
    category_name: str


@dataclass(frozen=True)
class SeriesRecord:
    """A contiguous daily store-SKU history."""

    store_index: int
    sku_index: int
    start_offset: int
    n_days: int


@dataclass(frozen=True)
class DatasetPlan:
    """Catalog and series windows for a profile. Demand is not simulated yet."""

    profile: str
    seed: int
    date_min: date
    date_max: date
    stores: tuple[StoreRecord, ...]
    skus: tuple[SkuRecord, ...]
    categories: tuple[CategoryRecord, ...]
    series: tuple[SeriesRecord, ...]
    row_count: int

    @property
    def observation_date_min(self) -> date:
        """Earliest day that appears on any series."""

        offset = min(item.start_offset for item in self.series)
        return self.date_min + timedelta(days=offset)

    @property
    def observation_date_max(self) -> date:
        """Latest day that appears on any series."""

        return self.date_max


@dataclass(frozen=True)
class _Calendar:
    epoch_day: I32
    dow: I32
    doy: I32
    holiday: BoolArray


@dataclass(frozen=True)
class _SeriesChunk:
    epoch_day: I32
    store_index: I32
    sku_index: I32
    units_sold: I64
    price: F64
    promotion: BoolArray
    discount_pct: F64
    stock_available: I64
    stockout: BoolArray
    holiday: BoolArray
    lead_time: I64
    weather_index: F64
    campaign_code: I32


def profile_names() -> tuple[str, ...]:
    """Return the supported generator profile names."""

    return tuple(sorted(_PROFILES))


def build_plan(profile: str, seed: int) -> DatasetPlan:
    """Build the catalog and daily series windows for ``profile`` and ``seed``."""

    if seed < 0:
        raise ValueError("RANDOM_SEED must be a non-negative integer.")
    selected = _PROFILES.get(profile)
    if selected is None:
        known = ", ".join(profile_names())
        raise ValueError(f"Unknown profile {profile!r}. Known profiles: {known}.")
    return _plan(selected, seed)


def generate_dataset(
    output: Path,
    *,
    profile: str,
    seed: int,
    max_rows: int,
) -> DatasetSummary:
    """Write the fact table, dimensions, and summary for one profile."""

    if max_rows <= 0:
        raise ValueError("MAX_DATASET_ROWS_DEMO must be a positive integer.")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path {output} is a file. Pass a directory.")

    plan = build_plan(profile, seed)
    if plan.row_count > max_rows:
        raise ValueError(
            f"Profile {profile!r} produces {plan.row_count} rows, "
            f"which is above MAX_DATASET_ROWS_DEMO ({max_rows})."
        )

    observations = _observations(plan)
    stores = _store_table(plan)
    skus = _sku_table(plan)
    categories = _category_table(plan)
    _require_contract(observations, stores, skus, categories)
    summary = _summary(plan, observations)

    output.mkdir(parents=True, exist_ok=True)
    _write_frame(observations, output / "observations.parquet", output / "observations.csv")
    _write_frame(stores, output / "stores.parquet", output / "stores.csv")
    _write_frame(skus, output / "skus.parquet", output / "skus.csv")
    _write_frame(categories, output / "categories.parquet", output / "categories.csv")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _plan(profile: GenerationProfile, seed: int) -> DatasetPlan:
    templates = _CATEGORIES[: profile.n_categories]
    categories = tuple(
        CategoryRecord(category_id=item.category_id, category_name=item.category_name)
        for item in templates
    )
    lifecycles = _lifecycles(profile.n_skus, seed)
    stores = tuple(_store(index) for index in range(profile.n_stores))
    skus = tuple(
        _sku(index, templates[index % profile.n_categories], lifecycles[index])
        for index in range(profile.n_skus)
    )
    span_days = (profile.date_max - profile.date_min).days + 1
    series: list[SeriesRecord] = []
    for store in stores:
        for sku in skus:
            if not _carried(store.index, sku.index, stock_all=profile.stock_all):
                continue
            start_offset = _start_offset(sku.index, sku.product_lifecycle, span_days)
            series.append(
                SeriesRecord(
                    store_index=store.index,
                    sku_index=sku.index,
                    start_offset=start_offset,
                    n_days=span_days - start_offset,
                )
            )
    if not series:
        raise ValueError(f"Profile {profile.name!r} produced no series.")
    covered_stores = {item.store_index for item in series}
    covered_skus = {item.sku_index for item in series}
    if covered_stores != set(range(profile.n_stores)) or covered_skus != set(range(profile.n_skus)):
        raise ValueError(f"Profile {profile.name!r} left a store or SKU without observations.")
    ordered = tuple(sorted(series, key=lambda item: (item.store_index, item.sku_index)))
    return DatasetPlan(
        profile=profile.name,
        seed=seed,
        date_min=profile.date_min,
        date_max=profile.date_max,
        stores=stores,
        skus=skus,
        categories=categories,
        series=ordered,
        row_count=sum(item.n_days for item in ordered),
    )


def _store(index: int) -> StoreRecord:
    region_index = index % len(_REGIONS)
    return StoreRecord(
        index=index,
        store_id=f"store-{index + 1:02d}",
        store_region=_REGIONS[region_index],
        scale=0.7 + (index % 5) * 0.15,
        region_phase=_REGION_PHASE[region_index],
    )


def _sku(index: int, template: _CategoryTemplate, lifecycle: str) -> SkuRecord:
    cents = template.base_cents + (index % 5) * 25
    return SkuRecord(
        index=index,
        sku_id=f"sku-{index + 1:03d}",
        category_id=template.category_id,
        product_brand=_BRANDS[index % len(_BRANDS)],
        product_lifecycle=lifecycle,
        list_price=cents / 100.0,
        base_demand=template.base_demand * (0.75 + (index % 6) * 0.1),
        weekly_amplitude=template.weekly_amplitude,
        annual_amplitude=template.annual_amplitude,
        lead_time_days=template.lead_time_days,
    )


def _lifecycles(n_skus: int, seed: int) -> list[str]:
    labels = [_MATURE] * n_skus
    if n_skus < 2:
        return labels
    n_new = 1 if n_skus < 8 else max(1, round(n_skus * 0.12))
    n_declining = 1 if n_skus < 8 else max(1, round(n_skus * 0.12))
    if n_new + n_declining > n_skus:
        n_declining = n_skus - n_new
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 101])))
    order = [int(index) for index in rng.permutation(n_skus)]
    for index in order[:n_new]:
        labels[index] = _NEW
    for index in order[n_new : n_new + n_declining]:
        labels[index] = _DECLINING
    return labels


def _carried(store_index: int, sku_index: int, *, stock_all: bool) -> bool:
    if stock_all:
        return True
    return (store_index + 2 * sku_index) % 3 != 0


def _start_offset(sku_index: int, lifecycle: str, span_days: int) -> int:
    if lifecycle != _NEW or span_days <= 1:
        return 0
    window = min(max(span_days // 5, 1), span_days // 2)
    offset = (sku_index * 17 + 3) % (window + 1)
    if offset == 0:
        offset = 1
    return min(offset, span_days - 1)


def _observations(plan: DatasetPlan) -> pa.Table:
    calendar = _calendar(plan.date_min, plan.date_max)
    chunks = [_series_chunk(plan, series, calendar) for series in plan.series]
    order = _order(chunks)
    epoch_day = _i32(_concat_i32([chunk.epoch_day for chunk in chunks])[order])
    store_index = _i32(_concat_i32([chunk.store_index for chunk in chunks])[order])
    sku_index = _i32(_concat_i32([chunk.sku_index for chunk in chunks])[order])
    units = _i64(_concat_i64([chunk.units_sold for chunk in chunks])[order])
    price = _f64(_concat_f64([chunk.price for chunk in chunks])[order])
    promotion = _bool(_concat_bool([chunk.promotion for chunk in chunks])[order])
    discount = _f64(_concat_f64([chunk.discount_pct for chunk in chunks])[order])
    stock = _i64(_concat_i64([chunk.stock_available for chunk in chunks])[order])
    stockout = _bool(_concat_bool([chunk.stockout for chunk in chunks])[order])
    holiday = _bool(_concat_bool([chunk.holiday for chunk in chunks])[order])
    lead_time = _i64(_concat_i64([chunk.lead_time for chunk in chunks])[order])
    weather = _f64(_concat_f64([chunk.weather_index for chunk in chunks])[order])
    campaign_code = _i32(_concat_i32([chunk.campaign_code for chunk in chunks])[order])

    store_ids = _take_strings([store.store_id for store in plan.stores], store_index)
    regions = _take_strings([store.store_region for store in plan.stores], store_index)
    sku_ids = _take_strings([sku.sku_id for sku in plan.skus], sku_index)
    category_ids = _take_strings([sku.category_id for sku in plan.skus], sku_index)
    brands = _take_strings([sku.product_brand for sku in plan.skus], sku_index)
    lifecycles = _take_strings([sku.product_lifecycle for sku in plan.skus], sku_index)
    campaigns = _take_strings(("", *_CAMPAIGNS), _i32(campaign_code + 1))

    arrays: list[pa.Array] = [
        pa.array(epoch_day, type=pa.date32()),
        store_ids,
        sku_ids,
        category_ids,
        pa.array(units, type=pa.int64()),
        pa.array(price, type=pa.float64()),
        pa.array(promotion, type=pa.bool_()),
        pa.array(discount, type=pa.float64()),
        pa.array(stock, type=pa.int64()),
        pa.array(stockout, type=pa.bool_()),
        pa.array(holiday, type=pa.bool_()),
        campaigns,
        pa.array(lead_time, type=pa.int64()),
        pa.array(weather, type=pa.float64()),
        regions,
        brands,
        lifecycles,
    ]
    table = pa.Table.from_arrays(arrays, schema=_arrow_schema(OBSERVATION_COLUMNS))
    return table.replace_schema_metadata(None)


def _order(chunks: Sequence[_SeriesChunk]) -> I64:
    epoch_day = _concat_i32([chunk.epoch_day for chunk in chunks])
    store_index = _concat_i32([chunk.store_index for chunk in chunks])
    sku_index = _concat_i32([chunk.sku_index for chunk in chunks])
    return np.lexsort((sku_index, store_index, epoch_day))


def _series_chunk(plan: DatasetPlan, series: SeriesRecord, calendar: _Calendar) -> _SeriesChunk:
    """Simulate one store-SKU history.

    Random draws stay in a fixed order: demand noise, then weather noise.
    Promotions, discounts, holidays, stock-outs, and anomalies follow the indices
    so the same seed rebuilds the same rows.
    """

    start = series.start_offset
    stop = start + series.n_days
    day_offset = _i32(np.arange(start, stop, dtype=np.int32))
    dow = _f64(calendar.dow[start:stop])
    doy = _f64(calendar.doy[start:stop])
    holiday = _bool(calendar.holiday[start:stop])
    store = plan.stores[series.store_index]
    sku = plan.skus[series.sku_index]
    n = series.n_days
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([plan.seed, store.index, sku.index]))
    )
    noise = _f64(rng.lognormal(mean=0.0, sigma=0.12, size=n))
    weather_noise = _f64(rng.normal(loc=0.0, scale=0.02, size=n))

    week = day_offset // 7
    promotion = _bool((store.index + 2 * sku.index + week) % 7 == 0)
    discount = _f64(np.where(promotion, _DISCOUNTS[(store.index + week) % 5], 0.0))
    anomaly = _bool((day_offset + store.index * 4 + sku.index * 6 + 17) % 47 == 0)
    stockout = _bool((day_offset + store.index * 5 + sku.index * 9) % 31 == 0)
    stockout = _bool(stockout & ~anomaly)

    weekly = 1.0 + sku.weekly_amplitude * np.sin(2.0 * np.pi * (dow - 5.0) / 7.0)
    annual = 1.0 + sku.annual_amplitude * np.cos(2.0 * np.pi * (doy - 350.0) / 365.25)
    price_effect = 1.0 + 0.8 * (discount / 100.0)
    holiday_effect = np.where(holiday, 1.45, 1.0)
    anomaly_effect = np.where(anomaly, 4.0, 1.0)
    expected = (
        sku.base_demand
        * store.scale
        * weekly
        * annual
        * _trend(sku.product_lifecycle, n)
        * price_effect
        * holiday_effect
        * noise
        * anomaly_effect
    )
    units = _i64(np.where(stockout, 0, np.rint(np.maximum(expected, 0.0))))
    buffer = np.int64(8 + (store.index + sku.index) % 12)
    stock = _i64(np.where(stockout, 0, units + buffer))
    price = _f64(np.round(sku.list_price * (1.0 - discount / 100.0), 2))
    seasonal = 0.5 + 0.35 * np.sin(2.0 * np.pi * (doy - store.region_phase) / 365.25)
    weather = _f64(np.clip(seasonal + weather_noise, 0.0, 1.0))
    campaign_code = _i32(np.where(promotion, (store.index + sku.index) % 4, -1))

    return _SeriesChunk(
        epoch_day=_i32(calendar.epoch_day[start:stop]),
        store_index=_i32(np.full(n, store.index, dtype=np.int32)),
        sku_index=_i32(np.full(n, sku.index, dtype=np.int32)),
        units_sold=units,
        price=price,
        promotion=promotion,
        discount_pct=discount,
        stock_available=stock,
        stockout=stockout,
        holiday=holiday,
        lead_time=_i64(np.full(n, sku.lead_time_days, dtype=np.int64)),
        weather_index=weather,
        campaign_code=campaign_code,
    )


def _trend(lifecycle: str, n: int) -> F64:
    age = np.arange(n, dtype=np.float64)
    if lifecycle == _DECLINING:
        return _f64(1.0 - 0.5 * (age / max(n - 1, 1)))
    if lifecycle == _NEW:
        return _f64(0.3 + 0.7 * np.minimum(age / 21.0, 1.0))
    return _f64(1.0 + 0.1 * (age / max(n - 1, 1)))


def _calendar(start: date, end: date) -> _Calendar:
    count = (end - start).days + 1
    holidays = _holiday_dates(start, end)
    epoch_day = np.empty(count, dtype=np.int32)
    dow = np.empty(count, dtype=np.int32)
    doy = np.empty(count, dtype=np.int32)
    holiday = np.zeros(count, dtype=np.bool_)
    for index in range(count):
        current = start + timedelta(days=index)
        epoch_day[index] = (current - _UNIX_EPOCH).days
        dow[index] = current.weekday()
        doy[index] = current.timetuple().tm_yday
        holiday[index] = current in holidays
    return _Calendar(
        epoch_day=_i32(epoch_day),
        dow=_i32(dow),
        doy=_i32(doy),
        holiday=_bool(holiday),
    )


def _holiday_dates(start: date, end: date) -> set[date]:
    found: set[date] = set()
    for year in range(start.year, end.year + 1):
        for month, day in _HOLIDAYS:
            current = date(year, month, day)
            if start <= current <= end:
                found.add(current)
    return found


def _store_table(plan: DatasetPlan) -> pa.Table:
    return _string_table(
        STORE_COLUMNS,
        [
            [store.store_id for store in plan.stores],
            [store.store_region for store in plan.stores],
        ],
    )


def _sku_table(plan: DatasetPlan) -> pa.Table:
    return _string_table(
        SKU_COLUMNS,
        [
            [sku.sku_id for sku in plan.skus],
            [sku.category_id for sku in plan.skus],
            [sku.product_brand for sku in plan.skus],
            [sku.product_lifecycle for sku in plan.skus],
        ],
    )


def _category_table(plan: DatasetPlan) -> pa.Table:
    return _string_table(
        CATEGORY_COLUMNS,
        [
            [category.category_id for category in plan.categories],
            [category.category_name for category in plan.categories],
        ],
    )


def _string_table(columns: Sequence[ColumnSpec], values: Sequence[Sequence[str]]) -> pa.Table:
    arrays = [pa.array(list(column), type=pa.string()) for column in values]
    table = pa.Table.from_arrays(arrays, schema=_arrow_schema(columns))
    return table.replace_schema_metadata(None)


def _summary(plan: DatasetPlan, observations: pa.Table) -> DatasetSummary:
    epoch_day = observations.column("date").cast(pa.int32()).to_numpy()
    date_min = _UNIX_EPOCH + timedelta(days=int(epoch_day.min()))
    date_max = _UNIX_EPOCH + timedelta(days=int(epoch_day.max()))
    if date_min != plan.observation_date_min or date_max != plan.observation_date_max:
        raise RuntimeError("Observation dates do not match the generation plan.")
    if observations.num_rows != plan.row_count:
        raise RuntimeError("Observation row count does not match the generation plan.")
    return DatasetSummary(
        categories=len(plan.categories),
        date_max=date_max.isoformat(),
        date_min=date_min.isoformat(),
        profile=plan.profile,
        row_count=plan.row_count,
        schema_version=SCHEMA_VERSION,
        seed=plan.seed,
        skus=len(plan.skus),
        stores=len(plan.stores),
    )


def _require_contract(
    observations: pa.Table,
    stores: pa.Table,
    skus: pa.Table,
    categories: pa.Table,
) -> None:
    checks = (
        ("observations", observation_schema_problems(observations.column_names, _logical_types(observations), exact=True)),
        ("stores", store_schema_problems(stores.column_names, _logical_types(stores), exact=True)),
        ("skus", sku_schema_problems(skus.column_names, _logical_types(skus), exact=True)),
        (
            "categories",
            category_schema_problems(categories.column_names, _logical_types(categories), exact=True),
        ),
    )
    problems = [f"{name}: {problem}" for name, found in checks for problem in found]
    if problems:
        detail = "\n".join(problems)
        raise RuntimeError(f"Generated tables do not match the retail demand contract.\n{detail}")


def _logical_types(table: pa.Table) -> dict[str, str]:
    return {name: _logical_type_name(table.schema.field(name).type) for name in table.column_names}


def _logical_type_name(data_type: pa.DataType) -> str:
    if pa.types.is_date32(data_type):
        return LogicalType.DATE.value
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return LogicalType.STRING.value
    if pa.types.is_int64(data_type):
        return LogicalType.INT64.value
    if pa.types.is_float64(data_type):
        return LogicalType.FLOAT64.value
    if pa.types.is_boolean(data_type):
        return LogicalType.BOOLEAN.value
    raise ValueError(f"Unsupported column type {data_type}.")


def _arrow_schema(columns: Sequence[ColumnSpec]) -> pa.Schema:
    return pa.schema([pa.field(column.name, _ARROW[column.logical_type]) for column in columns])


def _take_strings(values: Sequence[str], indices: I32) -> pa.Array:
    dictionary = pa.array(list(values), type=pa.string())
    taken = pc.take(dictionary, pa.array(indices, type=pa.int32()))
    return cast(pa.Array, taken)


def _write_frame(table: pa.Table, parquet_path: Path, csv_path: Path) -> None:
    pq.write_table(
        table,
        parquet_path,
        compression="snappy",
        version="2.6",
        data_page_version="1.0",
        write_statistics=True,
        use_dictionary=True,
        write_page_index=False,
    )
    pa_csv.write_csv(table, csv_path)


def _concat_i32(parts: Sequence[I32]) -> I32:
    return cast(I32, np.concatenate(list(parts)))


def _concat_i64(parts: Sequence[I64]) -> I64:
    return cast(I64, np.concatenate(list(parts)))


def _concat_f64(parts: Sequence[F64]) -> F64:
    return cast(F64, np.concatenate(list(parts)))


def _concat_bool(parts: Sequence[BoolArray]) -> BoolArray:
    return cast(BoolArray, np.concatenate(list(parts)))


def _i32(values: Any) -> I32:
    return cast(I32, np.array(values, dtype=np.int32, copy=True))


def _i64(values: Any) -> I64:
    return cast(I64, np.array(values, dtype=np.int64, copy=True))


def _f64(values: Any) -> F64:
    return cast(F64, np.array(values, dtype=np.float64, copy=True))


def _bool(values: Any) -> BoolArray:
    return cast(BoolArray, np.array(values, dtype=np.bool_, copy=True))

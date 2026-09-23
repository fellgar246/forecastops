"""Write a deterministic retail demand dataset.

Reads ``RANDOM_SEED`` and ``MAX_DATASET_ROWS_DEMO`` from the environment.
"""

import argparse
import os
import sys
from pathlib import Path

from forecastops_ml.data.synthetic import (
    DEFAULT_MAX_ROWS,
    DEFAULT_SEED,
    generate_dataset,
    profile_names,
)


def main(argv: list[str] | None = None) -> None:
    """Generate CSV, Parquet, and summary files for one profile."""

    parser = argparse.ArgumentParser(description="Generate a deterministic retail demand dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for the fact table, dimensions, and summary.json.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        choices=profile_names(),
        help="Catalog profile. The default profile is the full demo history.",
    )
    args = parser.parse_args(argv)
    try:
        summary = generate_dataset(
            args.output,
            profile=args.profile,
            seed=_env_int("RANDOM_SEED", DEFAULT_SEED, minimum=0),
            max_rows=_env_int("MAX_DATASET_ROWS_DEMO", DEFAULT_MAX_ROWS, minimum=1),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(
        f"Wrote {summary['row_count']} observations to {args.output} "
        f"(seed {summary['seed']}, {summary['date_min']} to {summary['date_max']})."
    )


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer.") from exc
    if value < minimum:
        raise SystemExit(f"{name} must be an integer greater than or equal to {minimum}.")
    return value


if __name__ == "__main__":
    main()

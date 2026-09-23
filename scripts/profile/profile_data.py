"""Write a demand profile for a generated dataset directory."""

import argparse
import sys
from pathlib import Path

from forecastops_ml.data.profile import profile_dataset


def main(argv: list[str] | None = None) -> None:
    """Read the dataset in ``--dataset`` and write ``profile.json`` beside it."""

    parser = argparse.ArgumentParser(description="Write profile.json for a generated dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Directory that contains observations and dimension tables.",
    )
    args = parser.parse_args(argv)
    try:
        destination = profile_dataset(args.dataset)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Wrote {destination}.")


if __name__ == "__main__":
    main()

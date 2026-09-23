"""Load the example environment before API tests import the application."""

import os

from envfile import ROOT, parse_env_file


def apply_example_env() -> None:
    """Copy `.env.example` into the process environment."""

    for key, value in parse_env_file(ROOT / ".env.example").items():
        os.environ[key] = value


apply_example_env()

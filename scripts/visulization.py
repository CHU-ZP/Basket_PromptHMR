"""Backward-compatible entrypoint for the misspelled visualization script.

Prefer running `python scripts/visualize_results.py ...` in new commands.
"""

import tyro

from visualize_results import main


if __name__ == "__main__":
    tyro.cli(main)

"""Backward-compatible entrypoint for the video splitting script.

Prefer running `python scripts/split_videos.py ...` in new commands.
"""

from scripts.split_videos import main


if __name__ == "__main__":
    main()

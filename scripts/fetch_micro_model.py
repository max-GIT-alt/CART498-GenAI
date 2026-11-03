"""Compatibility wrapper: download just the 2 GiB micro checkpoint.

Equivalent to ``python scripts/fetch_models.py --profile micro``.
"""
import sys

from fetch_models import main

if __name__ == "__main__":
    main(["--profile", "micro", *sys.argv[1:]])

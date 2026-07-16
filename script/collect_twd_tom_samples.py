#!/usr/bin/env python3
"""Compatibility wrapper for script/twd_tom/collect_samples.py."""

from script.twd_tom.collect_samples import *  # noqa: F401,F403
from script.twd_tom.collect_samples import main


if __name__ == "__main__":
    main()

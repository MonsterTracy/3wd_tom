#!/usr/bin/env python3
"""Compatibility wrapper for script/twd_tom/eval.py."""

from script.twd_tom.eval import *  # noqa: F401,F403
from script.twd_tom.eval import main


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Videos in, the agent's CSVs out. See pipeline/orchestrator.py (or --help) for the steps.

  python run_pipeline.py --videos "input/Day 1" --name d1 --vlm auto
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pipeline.orchestrator import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

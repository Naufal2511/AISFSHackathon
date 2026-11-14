#!/usr/bin/env python3
"""Herd size from anchor moments: the largest number of cows that are simultaneously CLEAR
(separated, fully visible, cameras agreeing) for at least --min-s seconds, over all runs of a
day. Resting periods (cows lying apart) provide these moments; feeding time rarely does.

  python scripts/herd_size.py outputs/level1_d1_1600 outputs/level1_d1_1950
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--min-s", type=int, default=10)
    a = ap.parse_args()
    best = 0
    for r in a.runs:
        p = os.path.join(r, "anchors", "anchor_score.csv")
        if not os.path.exists(p):
            print(f"{r}: no anchor_score.csv (run scripts/find_anchors.py)")
            continue
        df = pd.read_csv(p)
        k = max([k for k in range(1, int(df.clear.max()) + 1) if (df.clear >= k).sum() >= a.min_s], default=0)
        print(f"{r}: {int((df.clear >= k).sum())} s with >= {k} cows simultaneously clear (max ever {int(df.clear.max())})")
        best = max(best, k)
    print(f"HERD_SIZE={best}")


if __name__ == "__main__":
    main()

"""Tiny client for the Pen 6 HTTP API (requires `requests`).

    from src.api.client import Pen6Client
    c = Pen6Client("http://127.0.0.1:8800")            # token="..." when the server uses one
    runs = c.runs()
    obs = c.observations("level1_d1_1600_h", t0="2026-11-07T16:01:00", t1="2026-11-07T16:02:00")
    jpg = c.crop("level1_d1_1600_h", "cam1", "2026-11-07T16:01:10", cow_id="COW_03")   # bytes
    res = c.tool("level1_d1_1600_h", "best_view", cow_id="COW_03", time="16:01:10", purpose="head")
    c.submit_behavior("level1_d1_1600_h", predictions_df, model="beh-yolo-v1")
    c.submit_verdicts("level1_d1_1600_h", verdicts_df)
    act = c.activity("level1_d1_1600_h", stage="final")
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import requests


class Pen6Client:
    def __init__(self, base: str = "http://127.0.0.1:8800", token: Optional[str] = None, timeout: float = 300):
        self.base = base.rstrip("/")
        self.h = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout = timeout

    def _get(self, path, **params):
        r = requests.get(self.base + path, params={k: v for k, v in params.items() if v is not None},
                         headers=self.h, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        return r

    def _all(self, path, **params) -> pd.DataFrame:
        rows, offset = [], 0
        while True:
            d = self._get(path, offset=offset, limit=5000, **params).json()
            rows += d["rows"]
            offset += d["count"]
            if offset >= d["total"] or d["count"] == 0:
                return pd.DataFrame(rows)

    def runs(self):
        return self._get("/runs").json()["runs"]

    def manifest(self, run):
        return self._get(f"/runs/{run}/manifest").json()

    def cows(self, run):
        return self._get(f"/runs/{run}/cows").json()["cows"]

    def observations(self, run, t0=None, t1=None, cow_id=None, camera=None) -> pd.DataFrame:
        return self._all(f"/runs/{run}/observations", t0=t0, t1=t1, cow_id=cow_id, camera=camera)

    def cow_state(self, run, t0=None, t1=None, cow_id=None) -> pd.DataFrame:
        return self._all(f"/runs/{run}/cow_state", t0=t0, t1=t1, cow_id=cow_id)

    def frame(self, run, camera, t, boxes=False, width=1280) -> bytes:
        return self._get(f"/runs/{run}/frame.jpg", camera=camera, t=t, boxes=int(boxes), width=width).content

    def crop(self, run, camera, t, cow_id=None, track_id=None, pad=0.15, size=448) -> bytes:
        return self._get(f"/runs/{run}/crop.jpg", camera=camera, t=t, cow_id=cow_id, track_id=track_id,
                         pad=pad, size=size).content

    def tools(self, run):
        return self._get(f"/runs/{run}/tools").json()["tools"]

    def tool(self, run, name, **args):
        return self._get(f"/runs/{run}/tools/{name}",
                         **{k: (json.dumps(v) if not isinstance(v, str) else v) for k, v in args.items()}).json()

    def _post(self, path, df: pd.DataFrame, **params):
        r = requests.post(self.base + path, params=params, data=df.to_csv(index=False).encode(),
                          headers={**self.h, "Content-Type": "text/csv"}, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        return r.json()

    def submit_behavior(self, run, predictions: pd.DataFrame, model: Optional[str] = None):
        return self._post(f"/runs/{run}/behavior", predictions, **({"model": model} if model else {}))

    def submit_verdicts(self, run, verdicts: pd.DataFrame):
        return self._post(f"/runs/{run}/verdicts", verdicts)

    def activity(self, run, stage="best") -> pd.DataFrame:
        return self._all(f"/runs/{run}/activity", stage=stage)

    def time_budgets(self, run, stage="best") -> pd.DataFrame:
        return self._all(f"/runs/{run}/time_budgets", stage=stage)

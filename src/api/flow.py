"""Python API over one processed time slot ("run"), for the downstream behavior model and
validation agent. The HTTP server (src/api/server.py) is a thin layer over this class.

    from src.api.flow import Pen6Run
    run = Pen6Run("outputs/level1_d1_1600_h")
    obs = run.observations(t0="2026-11-07T16:01:00", t1="2026-11-07T16:02:00", camera="cam1")
    img = run.crop("cam1", "2026-11-07T16:01:10", cow_id="COW_03")          # BGR numpy array
    res = run.tool("best_view", cow_id="COW_03", time="16:01:10", purpose="head")
    run.submit_behavior(predictions_df, model="beh-yolo-v1")               # -> fused activity.csv
    run.submit_verdicts(verdicts_df)                                         # -> final activity.csv

Data contract: docs/DATA_CONTRACT.md. Times are wall-clock ISO strings (or epoch seconds).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from ..behavior.taxonomy import BEHAVIORS, NOT_VISIBLE

EPOCH = pd.Timestamp("1970-01-01")
P_COLS = ["p_lying", "p_standing", "p_walking", "p_feeding", "p_drinking", "p_rum_lying", "p_rum_standing",
          "p_grooming", "p_social", "p_other"]
P_TO_BEH = dict(zip(P_COLS, BEHAVIORS))
VIS_WEIGHT = {"visible": 1.0, "partial": 0.6}
KEYS = ["wall_time", "camera", "track_id", "track_piece"]


def to_sec(t) -> Optional[int]:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    if isinstance(t, (int, float, np.integer, np.floating)):
        return int(t)
    return int((pd.Timestamp(str(t).replace("T", " ")) - EPOCH).total_seconds())


def iso(sec) -> str:
    return (EPOCH + pd.to_timedelta(int(sec), unit="s")).strftime("%Y-%m-%dT%H:%M:%S")


class Pen6Run:
    def __init__(self, run_dir: str, video_root: str = "input"):
        self.dir = run_dir.rstrip("/")
        self.name = os.path.basename(self.dir)
        self.video_root = video_root
        self.contract = os.path.join(self.dir, "contract")
        if not os.path.exists(os.path.join(self.contract, "observations.csv")):
            raise FileNotFoundError(f"{self.dir} has no contract/observations.csv (run scripts/export_contract.py)")
        self._obs = self._state = None
        self._frames = self._toolbox = None

    # ------------------------------------------------------------------ read
    def manifest(self) -> Dict:
        m = json.load(open(os.path.join(self.contract, "manifest.json")))
        m["outputs"] = {k: os.path.exists(p) for k, p in self._paths().items()}
        return m

    def _paths(self):
        return {"behavior_predictions": os.path.join(self.contract, "behavior_predictions.csv"),
                "behavior_verdicts": os.path.join(self.contract, "behavior_verdicts.csv"),
                "activity_baseline": os.path.join(self.dir, "behavior", "activity.csv"),
                "activity_model": os.path.join(self.dir, "behavior_model", "activity.csv"),
                "activity_final": os.path.join(self.dir, "final", "activity.csv")}

    @property
    def obs(self) -> pd.DataFrame:
        if self._obs is None:
            o = pd.read_csv(os.path.join(self.contract, "observations.csv"), low_memory=False)
            o["sec"] = ((pd.to_datetime(o.wall_time) - EPOCH).dt.total_seconds()).astype(np.int64)
            self._obs = o
        return self._obs

    @property
    def state(self) -> pd.DataFrame:
        if self._state is None:
            s = pd.read_csv(os.path.join(self.contract, "cow_state.csv"), low_memory=False)
            s["sec"] = ((pd.to_datetime(s.wall_time) - EPOCH).dt.total_seconds()).astype(np.int64)
            self._state = s
        return self._state

    def cows(self) -> List[str]:
        return sorted(self.state.cow_id.dropna().unique().tolist())

    @staticmethod
    def _filter(df, t0=None, t1=None, cow_id=None, camera=None):
        if t0 is not None:
            df = df[df.sec >= to_sec(t0)]
        if t1 is not None:
            df = df[df.sec <= to_sec(t1)]
        if cow_id:
            df = df[df.cow_id == cow_id]
        if camera and "camera" in df:
            df = df[df.camera == camera]
        return df

    def observations(self, t0=None, t1=None, cow_id=None, camera=None) -> pd.DataFrame:
        return self._filter(self.obs, t0, t1, cow_id, camera).drop(columns="sec")

    def cow_state(self, t0=None, t1=None, cow_id=None) -> pd.DataFrame:
        return self._filter(self.state, t0, t1, cow_id).drop(columns="sec")

    # ------------------------------------------------------------------ images
    def _framestore(self):
        if self._frames is None:
            from ..agent.evidence import FrameStore
            from ..behavior.features import load_records
            rec = load_records([os.path.join(self.dir, "identity", "tracking_records_resolved.csv")])
            self._rec = rec
            self._frames = FrameStore(rec, self.video_root)
        return self._frames

    def frame(self, camera: str, t, boxes: bool = False) -> Optional[np.ndarray]:
        fs = self._framestore()
        sec = to_sec(t)
        fr = fs.frame(camera, float(sec))
        if fr is not None and boxes:
            from ..agent.evidence import draw_boxes
            r = self._rec
            fr = draw_boxes(fr, r[(r.camera_id == camera) & ((r.t - sec).abs() <= 0.6)])
        return fr

    def crop(self, camera: str, t, cow_id: Optional[str] = None, track_id: Optional[int] = None,
             pad: float = 0.15, size: int = 448) -> Optional[np.ndarray]:
        from ..agent.evidence import crop as crop_box
        sec = to_sec(t)
        o = self.obs[(self.obs.camera == camera) & (self.obs.sec == sec)]
        if cow_id:
            o = o[o.cow_id == cow_id]
        if track_id is not None:
            o = o[o.track_id == int(track_id)]
        if o.empty:
            return None
        b = o.iloc[0]
        fr = self.frame(camera, sec)
        return None if fr is None else crop_box(fr, (b.x1, b.y1, b.x2, b.y2), pad=pad, size=size)

    # ------------------------------------------------------------------ agent tools
    def _tools(self):
        if self._toolbox is None:
            from ..agent.evidence import Embedder
            from ..agent.memory import Memory
            from ..agent.tools import Toolbox
            from ..behavior.rumination import index_videos
            from ..state.world_state import WorldState
            fs = self._framestore()
            tl_path = os.path.join(self.dir, "behavior", "behavior_timeline_1s.csv")
            tl = None
            if os.path.exists(tl_path):
                tl = pd.read_csv(tl_path)
                tl["t"] = (pd.to_datetime(tl.wall_time) - EPOCH).dt.total_seconds()
            mem = Memory(os.path.join(self.dir, "agent", "memory"))
            cam_pos = {k: v.get("mount_position_estimated")
                       for k, v in yaml.safe_load(open("configs/cameras.yaml"))["cameras"].items()}
            self._toolbox = Toolbox(self._rec, fs, mem, tl, Embedder(), index_videos(self.video_root),
                                    world=WorldState.load(os.path.join(self.dir, "state")), cam_pos=cam_pos)
        return self._toolbox

    def tools(self) -> List[Dict]:
        return [{"name": t.name, "args": t.args, "description": t.description, "evidence": t.evidence}
                for t in self._tools().tools.values()]

    def tool(self, name: str, **args):
        """Run one agent tool (same as the supervisor agent uses). Returns ToolResult(text, images, data)."""
        return self._tools().call(name, args)

    # ------------------------------------------------------------------ behavior model in
    def submit_behavior(self, preds: Union[pd.DataFrame, str], model: Optional[str] = None) -> Dict:
        """Per-box predictions (observations.csv rows with p_* filled, or at least the key columns
        wall_time, camera, track_id[, track_piece] + p_*). Fuses them per cow per second and writes
        <run>/behavior_model/ (activity.csv, time_budgets.csv, behavior_timeline_1s.csv, ...)."""
        p = pd.read_csv(preds) if isinstance(preds, str) else preds.copy()
        missing = [c for c in P_COLS if c not in p.columns]
        if missing:
            raise ValueError(f"missing probability columns: {missing} (see docs/DATA_CONTRACT.md)")
        if "track_piece" not in p:
            p["track_piece"] = 0
        # accept "2026-11-07 16:00:30", "2026-11-07T16:00:30" or epoch seconds
        wt = p["wall_time"] if "wall_time" in p else p["sec"]
        wt = pd.to_datetime(wt, unit="s") if np.issubdtype(np.asarray(wt).dtype, np.number) else pd.to_datetime(wt)
        p["wall_time"] = wt.dt.strftime("%Y-%m-%dT%H:%M:%S")
        p[P_COLS] = p[P_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        s = p[P_COLS].sum(axis=1).replace(0, np.nan)
        p[P_COLS] = p[P_COLS].div(s, axis=0).fillna(0.0)
        if model:
            p["model"] = model
        p["behavior"] = p[P_COLS].idxmax(axis=1).map(P_TO_BEH)
        p["beh_conf"] = p[P_COLS].max(axis=1).round(3)
        o = self.obs.drop(columns=[c for c in P_COLS + ["behavior", "beh_conf", "model"] if c in self.obs])
        o["track_id"] = o.track_id.astype(int)
        o["track_piece"] = o.track_piece.fillna(0).astype(int)
        p["track_id"] = p.track_id.astype(int)
        p["track_piece"] = p.track_piece.fillna(0).astype(int)
        keep = KEYS + P_COLS + ["behavior", "beh_conf"] + (["model"] if "model" in p else [])
        m = o.merge(p[keep].drop_duplicates(KEYS), on=KEYS, how="left")
        matched = int(m.p_lying.notna().sum())
        if matched == 0:
            raise ValueError("no prediction row matched an observation (keys: wall_time, camera, track_id, track_piece)")
        os.makedirs(self.contract, exist_ok=True)
        m.drop(columns="sec").to_csv(os.path.join(self.contract, "behavior_predictions.csv"), index=False)
        tl = self._fuse(m)
        out = self._write_outputs(tl, os.path.join(self.dir, "behavior_model"))
        return {"matched_boxes": matched, "boxes": int(len(o)), "cow_seconds": int(len(tl)),
                "model": str(p["model"].iloc[0]) if "model" in p else None, **out}

    def _fuse(self, m: pd.DataFrame) -> pd.DataFrame:
        """Per cow per second: probability average over that cow's body boxes, weighted by
        detection confidence x visibility (visible 1.0, partial 0.6, other 0.8)."""
        b = m[m.cow_id.notna() & m.p_lying.notna() & (m.get("box_role", "body") == "body")].copy()
        b["w"] = b.det_conf.fillna(0.5) * b.visibility.map(VIS_WEIGHT).fillna(0.8)
        P = b[P_COLS].to_numpy() * b.w.to_numpy()[:, None]
        agg = pd.DataFrame(P, columns=P_COLS, index=b.index).join(b[["cow_id", "sec", "w", "camera"]])
        g = agg.groupby(["cow_id", "sec"])
        fused = g[P_COLS].sum().div(g.w.sum().replace(0, np.nan), axis=0).fillna(0.0)
        fused["cameras"] = g.camera.agg(lambda c: ",".join(sorted(set(c))))
        fused = fused.reset_index()
        fused["behavior"] = fused[P_COLS].idxmax(axis=1).map(P_TO_BEH)
        fused["confidence"] = fused[P_COLS].max(axis=1)
        fused["evidence"], fused["observed"] = "observed", True
        # every cow-second the world state knows about but no box predicted -> not visible
        allcs = self.state[["cow_id", "sec"]].drop_duplicates()
        tl = allcs.merge(fused, on=["cow_id", "sec"], how="left")
        nv = tl.behavior.isna()
        tl.loc[nv, "behavior"], tl.loc[nv, "evidence"] = NOT_VISIBLE, "not_visible"
        tl.loc[nv, "confidence"], tl.loc[nv, "observed"] = 0.0, False
        tl[P_COLS] = tl[P_COLS].fillna(0.0)
        tl.loc[nv, "p_other"] = 1.0
        tl = tl.rename(columns={"cow_id": "global_cow_id", **{c: f"p::{P_TO_BEH[c]}" for c in P_COLS}})
        tl["observed"] = tl.observed.astype(bool)
        return tl.sort_values(["global_cow_id", "sec"]).reset_index(drop=True)

    def _write_outputs(self, tl: pd.DataFrame, out_dir: str) -> Dict:
        from ..behavior.export import write_all
        cfg = yaml.safe_load(open("configs/behavior.yaml"))
        paths = write_all(out_dir, tl, cfg)
        return {"outputs": {k: v for k, v in paths.items()}}

    # ------------------------------------------------------------------ validation agent in
    def submit_verdicts(self, verdicts: Union[pd.DataFrame, str], base: str = "auto") -> Dict:
        """Validation-agent verdicts (cow_id, t0, t1, behavior_in, verdict, behavior_out, confidence,
        evidence, trace_file). Applied on top of the behavior model's fused timeline (or our
        baseline when no model output exists) -> <run>/final/ activity.csv, time_budgets.csv."""
        v = pd.read_csv(verdicts) if isinstance(verdicts, str) else verdicts.copy()
        need = ["cow_id", "t0", "t1", "verdict"]
        miss = [c for c in need if c not in v]
        if miss:
            raise ValueError(f"missing verdict columns: {miss}")
        bad = v[~v.verdict.isin(["confirm", "override", "uncertain"])]
        if len(bad):
            raise ValueError("verdict must be confirm / override / uncertain")
        ov = v[v.verdict == "override"]
        if len(ov) and ("behavior_out" not in v or (~ov.behavior_out.isin(BEHAVIORS)).any()):
            raise ValueError(f"override needs behavior_out in {BEHAVIORS}")
        src_dir = os.path.join(self.dir, "behavior_model")
        if base == "baseline" or (base == "auto" and not os.path.exists(os.path.join(src_dir, "behavior_timeline_1s.csv"))):
            src_dir = os.path.join(self.dir, "behavior")
        tl = pd.read_csv(os.path.join(src_dir, "behavior_timeline_1s.csv")).rename(columns={"cow_id": "global_cow_id"})
        tl["sec"] = ((pd.to_datetime(tl.wall_time) - EPOCH).dt.total_seconds()).astype(np.int64)
        tl = tl.drop(columns="wall_time")
        tl["observed"] = tl.observed.astype(bool)
        changed = 0
        for r in v.itertuples():
            m = (tl.global_cow_id == r.cow_id) & tl.sec.between(to_sec(r.t0), to_sec(r.t1))
            if r.verdict == "override":
                tl.loc[m, "behavior"] = r.behavior_out
                tl.loc[m, "evidence"] = "agent_override"
                if hasattr(r, "confidence") and pd.notna(r.confidence):
                    tl.loc[m, "confidence"] = float(r.confidence)
                changed += int(m.sum())
            elif r.verdict == "confirm":
                tl.loc[m, "evidence"] = "agent_confirmed"
            else:
                tl.loc[m, "evidence"] = "agent_uncertain"
        v.to_csv(os.path.join(self.contract, "behavior_verdicts.csv"), index=False)
        out = self._write_outputs(tl, os.path.join(self.dir, "final"))
        return {"verdicts": int(len(v)), "seconds_changed": changed, "based_on": os.path.relpath(src_dir, self.dir), **out}

    # ------------------------------------------------------------------ results
    def activity(self, stage: str = "best") -> pd.DataFrame:
        p = self._paths()
        order = {"final": ["activity_final"], "model": ["activity_model"], "baseline": ["activity_baseline"],
                 "best": ["activity_final", "activity_model", "activity_baseline"]}[stage]
        for k in order:
            if os.path.exists(p[k]):
                return pd.read_csv(p[k]).assign(stage=k.replace("activity_", ""))
        raise FileNotFoundError(f"no activity.csv for stage {stage}")

    def time_budgets(self, stage: str = "best") -> pd.DataFrame:
        sub = {"final": ["final"], "model": ["behavior_model"], "baseline": ["behavior"],
               "best": ["final", "behavior_model", "behavior"]}[stage]
        for s in sub:
            f = os.path.join(self.dir, s, "time_budgets.csv")
            if os.path.exists(f):
                return pd.read_csv(f).assign(stage=s)
        raise FileNotFoundError(f"no time_budgets.csv for stage {stage}")


def list_runs(root: str = "outputs") -> List[str]:
    return sorted(d for d in os.listdir(root)
                  if os.path.exists(os.path.join(root, d, "contract", "observations.csv")))

"""Top-view world state of every cow, second by second, plus per-camera visibility.

Built from the identity-resolved tracking records (one row per box):

  cow_state.csv        per cow per second: fused position (Kalman, all cameras), predicted
                       position when nobody sees it (+ uncertainty), velocity, speed, heading,
                       how many cameras observed it, the camera that saw it best
  visibility.csv       per cow per camera per second: visible | partial | occluded | not_detected
                       | out_of_view, with the occluding cow, box size and facing
                       (toward / away / side-on) relative to that camera
  visibility_intervals.csv   the same compressed into intervals (what the VLM tools read)
  anomalies.csv        jumps (observation far from the prediction), camera conflicts,
                       impossible speeds, positions outside the pen
  heatmap.npz          occupancy per cow on a 0.5 m grid

Why: the agent should look only at views that can answer its question (unoccluded, head
toward the camera, close), and an impossible move should be caught by physics, not by eye.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

DEFAULTS = dict(
    pen=(12.0, 30.0),
    sigma_m={"cam1": 0.6, "cam2": 0.5, "cam3": 0.8},   # base measurement noise per camera
    sigma_per_m=0.03,                                   # + noise per metre of camera distance
    accel=0.6,                                          # process noise (m/s^2)
    gate_chi2=13.8,                                     # 2-dof chi-square 99.9%
    jump_m=3.0,                                         # and the miss must be at least this big
    conflict_m=3.0,
    max_speed=3.0,
    heading_min_speed=0.25,
    heading_hold_s=60,
    partial_iou=0.2,
    edge_frac=0.01,
    occlusion_status_partial=0.35,
    grid=0.5,
)


def _bearing(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _facing(heading, cow_xy, cam_xy):
    if heading is None or cam_xy is None:
        return "unknown"
    to_cam = _bearing(cow_xy, cam_xy)
    d = abs((heading - to_cam + math.pi) % (2 * math.pi) - math.pi)
    return "toward" if d < math.radians(50) else "away" if d > math.radians(130) else "side-on"


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class WorldState:
    def __init__(self, transformer, cam_pos: Dict[str, tuple], boundary=None, cfg: Optional[Dict] = None):
        self.tr, self.cam_pos, self.boundary = transformer, cam_pos, boundary
        self.c = {**DEFAULTS, **(cfg or {})}

    # ------------------------------------------------------------------ fusion
    def _R(self, cam, xy):
        cp = self.cam_pos.get(cam)
        dist = math.hypot(xy[0] - cp[0], xy[1] - cp[1]) if cp else 10.0
        s = self.c["sigma_m"].get(cam, 0.8) + self.c["sigma_per_m"] * dist
        return np.eye(2) * s * s

    def build(self, rec: pd.DataFrame):
        c = self.c
        rec = rec.copy()
        rec["sec"] = np.round(rec.t).astype(np.int64)
        idr = rec[rec.global_cow_id.notna() & rec.world_x.notna()]
        t_lo, t_hi = int(rec.sec.min()), int(rec.sec.max())
        obs = {k: g for k, g in idr.groupby(["global_cow_id", "sec"])}
        cows = sorted(idr.global_cow_id.unique())
        F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        q = c["accel"] ** 2
        Q = q * np.array([[.25, 0, .5, 0], [0, .25, 0, .5], [.5, 0, 1, 0], [0, .5, 0, 1]])
        Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        states, anomalies = [], []
        for cid in cows:
            x, P, last_obs, heading, heading_t = None, None, None, None, None
            for sec in range(t_lo, t_hi + 1):
                if x is not None:
                    x = F @ x
                    P = F @ P @ F.T + Q
                g = obs.get((cid, sec))
                n_obs, misses, best_cam = 0, [], None
                if g is not None:
                    meas = g.groupby("camera_id")[["world_x", "world_y"]].mean()
                    if len(meas) >= 2:
                        M = meas.to_numpy()
                        spread = float(np.max(np.linalg.norm(M[:, None] - M[None], axis=2)))
                        if spread > c["conflict_m"]:
                            where = ", ".join(f"{k} ({v.world_x:.1f},{v.world_y:.1f})" for k, v in meas.iterrows())
                            odd = None
                            if len(meas) >= 3:  # two cameras agree, one is off -> calibration offset, not identity
                                for k in meas.index:
                                    rest = meas.drop(k).to_numpy()
                                    if np.max(np.linalg.norm(rest[:, None] - rest[None], axis=2)) < 1.5:
                                        odd = k
                            anomalies.append({"cow_id": cid, "sec": sec,
                                              "type": "camera_offset" if odd else "camera_conflict",
                                              "detail": (f"{odd} places it {spread:.1f} m from the other cameras: " if odd
                                                         else f"cameras disagree by {spread:.1f} m: ") + where})
                    areas = g.assign(a=(g.bbox_x2 - g.bbox_x1) * (g.bbox_y2 - g.bbox_y1) / (g.frame_width * g.frame_height))
                    best_cam = areas.sort_values("a").camera_id.iloc[-1]
                    for cam, m in meas.iterrows():
                        z = m.to_numpy(float)
                        R = self._R(cam, z)
                        if x is None:
                            x = np.r_[z, 0, 0]
                            P = np.diag([R[0, 0], R[1, 1], 1.0, 1.0])
                            n_obs += 1
                            continue
                        y = z - Hm @ x
                        S = Hm @ P @ Hm.T + R
                        d2 = float(y @ np.linalg.solve(S, y))
                        if d2 > c["gate_chi2"] and np.linalg.norm(y) > c["jump_m"]:
                            misses.append((cam, float(np.linalg.norm(y)), z))
                            continue
                        K = P @ Hm.T @ np.linalg.inv(S)
                        x = x + K @ y
                        P = (np.eye(4) - K @ Hm) @ P
                        n_obs += 1
                    if misses and n_obs == 0:  # every camera says the cow is somewhere else: a jump
                        cam, dist, z = max(misses, key=lambda m: m[1])
                        anomalies.append({"cow_id": cid, "sec": sec, "type": "jump",
                                          "detail": f"seen {dist:.1f} m from where it should be ({cam}); "
                                                    f"predicted ({x[0]:.1f},{x[1]:.1f}) observed ({z[0]:.1f},{z[1]:.1f})"})
                        x = np.r_[z, 0, 0]  # re-anchor on the observation (the ID may have switched)
                        P = np.diag([1.0, 1.0, 1.0, 1.0])
                        n_obs = len(misses)
                if x is None:
                    continue
                speed = float(math.hypot(x[2], x[3]))
                if n_obs and speed >= c["heading_min_speed"]:
                    heading, heading_t = math.atan2(x[3], x[2]), sec
                if heading_t is not None and sec - heading_t > c["heading_hold_s"]:
                    heading = None
                if n_obs and speed > c["max_speed"]:
                    anomalies.append({"cow_id": cid, "sec": sec, "type": "speed", "detail": f"{speed:.1f} m/s"})
                W, L = c["pen"]
                if n_obs and not (-1.0 <= x[0] <= W + 1.0 and -1.5 <= x[1] <= L + 1.0):
                    anomalies.append({"cow_id": cid, "sec": sec, "type": "outside_pen",
                                      "detail": f"at ({x[0]:.1f},{x[1]:.1f})"})
                states.append({"cow_id": cid, "sec": sec, "x": round(float(x[0]), 2), "y": round(float(x[1]), 2),
                               "vx": round(float(x[2]), 2), "vy": round(float(x[3]), 2), "speed": round(speed, 2),
                               "heading_deg": None if heading is None else round(math.degrees(heading), 0),
                               "pos_sd_m": round(float(math.sqrt(max(P[0, 0], P[1, 1]))), 2),
                               "n_cams": int(n_obs), "observed": bool(n_obs), "best_cam": best_cam})
        self.state = pd.DataFrame(states)
        self.anomalies = self._group_anomalies(pd.DataFrame(anomalies, columns=["cow_id", "sec", "type", "detail"]))
        self.visibility = self._visibility(rec)
        self.intervals = self._intervals(self.visibility)
        self.heat = self._heatmap()
        return self

    @staticmethod
    def _group_anomalies(a: pd.DataFrame) -> pd.DataFrame:
        if a.empty:
            return pd.DataFrame(columns=["cow_id", "type", "t0", "t1", "seconds", "detail"])
        out = []
        for (cid, typ), g in a.sort_values("sec").groupby(["cow_id", "type"]):
            run = None
            for r in g.itertuples():
                if run and r.sec - run["t1"] <= 2:
                    run["t1"], run["seconds"] = r.sec, run["seconds"] + 1
                else:
                    if run:
                        out.append(run)
                    run = {"cow_id": cid, "type": typ, "t0": r.sec, "t1": r.sec, "seconds": 1, "detail": r.detail}
            out.append(run)
        return pd.DataFrame(out).sort_values("t0").reset_index(drop=True)

    # ------------------------------------------------------------------ visibility
    def _visibility(self, rec: pd.DataFrame) -> pd.DataFrame:
        c = self.c
        cams = sorted(rec.camera_id.unique())
        size = {cam: (int(g.frame_width.iloc[0]), int(g.frame_height.iloc[0])) for cam, g in rec.groupby("camera_id")}
        by_cs = {k: g for k, g in rec.groupby(["camera_id", "sec"])}
        st = self.state.set_index(["cow_id", "sec"])
        out = []
        for (cid, sec), s in st.iterrows():
            heading = None if pd.isna(s.heading_deg) else math.radians(s.heading_deg)
            for cam in cams:
                W, H = size[cam]
                g = by_cs.get((cam, sec))
                face = _facing(heading, (s.x, s.y), self.cam_pos.get(cam))
                row = {"cow_id": cid, "sec": sec, "camera": cam, "facing": face, "status": None,
                       "track": None, "box_area": None, "occluder": None}
                mine = g[g.global_cow_id == cid] if g is not None else None
                if mine is not None and len(mine):
                    b = mine.iloc[0]
                    bb = (b.bbox_x1, b.bbox_y1, b.bbox_x2, b.bbox_y2)
                    edge = (bb[0] < c["edge_frac"] * W or bb[2] > (1 - c["edge_frac"]) * W or
                            bb[1] < c["edge_frac"] * H or bb[3] > (1 - c["edge_frac"]) * H)
                    overlap, occl = 0.0, None
                    for o in g.itertuples():
                        if o.local_track_id == b.local_track_id:
                            continue
                        v = _iou(bb, (o.bbox_x1, o.bbox_y1, o.bbox_x2, o.bbox_y2))
                        if v > overlap:
                            overlap, occl = v, o.global_cow_id if isinstance(o.global_cow_id, str) else f"track {o.local_track_id}"
                    occ_s = pd.to_numeric(b.occlusion_status, errors="coerce")
                    partial = edge or overlap > c["partial_iou"] or (pd.notna(occ_s) and occ_s > c["occlusion_status_partial"])
                    row.update(status="partial" if partial else "visible", track=int(b.local_track_id),
                               box_area=round(float((bb[2] - bb[0]) * (bb[3] - bb[1]) / (W * H)), 4),
                               occluder=occl if partial and overlap > c["partial_iou"] else ("frame edge" if edge else None))
                else:
                    with np.errstate(all="ignore"):
                        p = self.tr.world_to_pixel(cam, float(s.x), float(s.y), current_res=(W, H))
                    if p is not None and not (np.isfinite(p[0]) and np.isfinite(p[1])):
                        p = None
                    in_frame = p is not None and 0 <= p[0] < W and 0 <= p[1] < H
                    if in_frame and self.boundary is not None:
                        in_frame = self.boundary.is_point_inside(cam, p, W, H, tolerance_px=0.03 * W)
                    if not in_frame:
                        row["status"] = "out_of_view"
                    else:
                        occl = None
                        if g is not None:
                            for o in g.itertuples():
                                pad = 0.1 * (o.bbox_x2 - o.bbox_x1)
                                if o.bbox_x1 - pad <= p[0] <= o.bbox_x2 + pad and o.bbox_y1 - pad <= p[1] <= o.bbox_y2 + pad:
                                    occl = o.global_cow_id if isinstance(o.global_cow_id, str) else f"track {o.local_track_id}"
                                    break
                        row["status"] = "occluded" if occl else "not_detected"
                        row["occluder"] = occl
                out.append(row)
        return pd.DataFrame(out)

    @staticmethod
    def _intervals(v: pd.DataFrame) -> pd.DataFrame:
        out = []
        for (cid, cam), g in v.sort_values("sec").groupby(["cow_id", "camera"]):
            run = None
            for r in g.itertuples():
                key = (r.status, r.occluder if r.status in ("occluded", "partial") else None)
                if run and run["_k"] == key and r.sec - run["t1"] <= 1:
                    run["t1"] = r.sec
                    run["facing"].append(r.facing)
                else:
                    if run:
                        out.append(run)
                    run = {"cow_id": cid, "camera": cam, "status": r.status, "occluder": key[1], "t0": r.sec,
                           "t1": r.sec, "facing": [r.facing], "_k": key}
            if run:
                out.append(run)
        for r in out:
            f = [x for x in r.pop("facing") if x != "unknown"]
            r["facing"] = max(set(f), key=f.count) if f else "unknown"
            r.pop("_k")
            r["seconds"] = r["t1"] - r["t0"] + 1
        return pd.DataFrame(out)

    def _heatmap(self):
        W, L = self.c["pen"]
        gsz = self.c["grid"]
        nx, ny = int(W / gsz), int(L / gsz)
        heat = {}
        for cid, g in self.state[self.state.observed].groupby("cow_id"):
            h, _, _ = np.histogram2d(g.x.clip(0, W - 1e-3), g.y.clip(0, L - 1e-3), bins=[nx, ny], range=[[0, W], [0, L]])
            heat[cid] = h
        return heat

    # ------------------------------------------------------------------ queries (used by agent tools)
    def best_view(self, cid: str, sec: int, purpose: str = "identity", window: int = 5) -> Optional[Dict]:
        v = self.visibility
        g = v[(v.cow_id == cid) & v.sec.between(sec - window, sec + window) & v.status.isin(["visible", "partial"])]
        if g.empty:
            return None
        score = g.box_area.fillna(0) * np.where(g.status == "visible", 1.0, 0.4)
        if purpose == "head":
            score = score * np.where(g.facing == "toward", 2.0, np.where(g.facing == "side-on", 0.8, 0.2))
        elif purpose == "identity":
            score = score * np.where(g.facing == "side-on", 1.5, 1.0)
        score = score / (1 + 0.05 * (g.sec - sec).abs())
        b = g.loc[score.idxmax()]
        return {"camera": b.camera, "sec": int(b.sec), "track": None if pd.isna(b.track) else int(b.track),
                "status": b.status, "facing": b.facing, "box_area": b.box_area}

    def save(self, out_dir: str):
        import os
        os.makedirs(out_dir, exist_ok=True)
        self.state.to_csv(os.path.join(out_dir, "cow_state.csv"), index=False)
        self.visibility.to_csv(os.path.join(out_dir, "visibility.csv"), index=False)
        self.intervals.to_csv(os.path.join(out_dir, "visibility_intervals.csv"), index=False)
        self.anomalies.to_csv(os.path.join(out_dir, "anomalies.csv"), index=False)
        np.savez(os.path.join(out_dir, "heatmap.npz"), **self.heat)

    @classmethod
    def load(cls, out_dir: str):
        import os
        ws = cls.__new__(cls)
        ws.c = dict(DEFAULTS)
        ws.state = pd.read_csv(os.path.join(out_dir, "cow_state.csv"))
        ws.visibility = pd.read_csv(os.path.join(out_dir, "visibility.csv"))
        ws.intervals = pd.read_csv(os.path.join(out_dir, "visibility_intervals.csv"))
        ws.anomalies = pd.read_csv(os.path.join(out_dir, "anomalies.csv"))
        z = np.load(os.path.join(out_dir, "heatmap.npz"))
        ws.heat = {k: z[k] for k in z.files}
        return ws


def render_top_view(state: pd.DataFrame, cows, t0: int, t1: int, heat: Optional[np.ndarray] = None,
                    anomalies: Optional[pd.DataFrame] = None, cam_pos=None, scale: int = 24):
    """Top-view image of the pen: occupancy heatmap + each cow's path (front/bunk at the bottom)."""
    import cv2
    W, L = 12, 30
    pad = 40
    img = np.full((L * scale + 2 * pad, W * scale + 2 * pad, 3), 245, np.uint8)
    P = lambda x, y: (int(pad + x * scale), int(pad + (L - y) * scale))
    if heat is not None and heat.max() > 0:
        h = (255 * (heat / heat.max()) ** 0.5).astype(np.uint8).T[::-1]
        hm = cv2.applyColorMap(cv2.resize(h, (W * scale, L * scale), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)
        mask = cv2.resize(h, (W * scale, L * scale), interpolation=cv2.INTER_NEAREST) > 0
        roi = img[pad:pad + L * scale, pad:pad + W * scale]
        roi[mask] = (0.55 * hm[mask] + 0.45 * roi[mask]).astype(np.uint8)
    cv2.rectangle(img, P(0, 30), P(12, 0), (90, 90, 90), 2)
    cv2.rectangle(img, P(0, 0), P(12, -1.0), (100, 160, 200), -1)
    cv2.putText(img, "feed bunk (front)", P(3.2, -0.7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    cv2.putText(img, "chute side", P(-1.4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
    for cam, p in (cam_pos or {}).items():
        if p:
            cv2.circle(img, P(p[0], p[1]), 5, (200, 80, 0), -1)
            cv2.putText(img, cam, (P(p[0], p[1])[0] + 6, P(p[0], p[1])[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 80, 0), 1)
    palette = [(75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0), (48, 130, 245), (180, 30, 145),
               (240, 240, 70), (230, 50, 240), (60, 245, 210), (128, 128, 0)]
    for i, cid in enumerate(cows):
        g = state[(state.cow_id == cid) & state.sec.between(t0, t1)].sort_values("sec")
        if g.empty:
            continue
        col = palette[i % len(palette)]
        pts = np.array([P(x, y) for x, y in zip(g.x, g.y)], np.int32)
        for (a, b), obs in zip(zip(pts[:-1], pts[1:]), g.observed.to_numpy()[1:]):
            cv2.line(img, tuple(a), tuple(b), col, 2 if obs else 1, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 6, col, -1)
        cv2.putText(img, cid.replace("COW_", ""), (pts[-1][0] + 7, pts[-1][1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
    if anomalies is not None and len(anomalies):
        st = state.set_index(["cow_id", "sec"])
        for r in anomalies[anomalies.t0.between(t0, t1) & anomalies.cow_id.isin(cows)].itertuples():
            if (r.cow_id, r.t0) in st.index:
                s = st.loc[(r.cow_id, r.t0)]
                cv2.drawMarker(img, P(s.x, s.y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    return img

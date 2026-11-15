"""30-minute ID-check sheets (a Level 1 deliverable).

One row per cow: memory-card reference crops | what each camera shows under that ID at the
audit time (nearest box within +-30 s) | appearance similarity of that box to the card.
Saved as outputs/agent/id_checks/id_check_<YYYYmmdd_HHMMSS>.jpg plus a CSV row per cow.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from .evidence import crop, label, tile
from .memory import Memory


def _fit(im: np.ndarray, cell: int) -> np.ndarray:
    """Letterbox a crop into a cell x cell square, so the label drawn on top is never cut off by a narrow crop."""
    s = cell / max(im.shape[:2])
    r = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
    out = np.zeros((cell, cell, 3), np.uint8)
    y, x = (cell - r.shape[0]) // 2, (cell - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def build_sheet(t: float, rec: pd.DataFrame, frames, memory: Memory, embedder, out_dir: str,
                cameras=("cam1", "cam2", "cam3"), tol: float = 30.0) -> Tuple[str, List[Dict], np.ndarray]:
    os.makedirs(out_dir, exist_ok=True)
    rows_img, rows = [], []
    cell = 200
    near = rec[(rec.t - t).abs() <= tol]
    cache = {}
    for cid in memory.known_ids():
        ref = memory.reference_strip(cid, n=2, cell=cell)[:cell]
        cells = [label(ref, f"{cid} card")]
        row = {"time": str(pd.Timestamp(t, unit="s")), "cow_id": cid}
        for cam in cameras:
            g = near[(near.global_cow_id == cid) & (near.camera_id == cam)]
            if g.empty:
                cells.append(label(np.zeros((cell, cell, 3), np.uint8), f"{cam}: not seen"))
                row[f"{cam}_seen"] = False
                continue
            r = g.loc[(g.t - t).abs().idxmin()]
            key = (cam, round(float(r.t), 1))
            if key not in cache:
                cache[key] = frames.frame(cam, float(r.t))
            fr = cache[key]
            if fr is None:
                cells.append(label(np.zeros((cell, cell, 3), np.uint8), f"{cam}: no video"))
                continue
            bb = (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2)
            sim = memory.similarity(embedder(fr, bb)) if embedder is not None else []
            own = next((s for c, s in sim if c == cid), None)
            best = sim[0] if sim else (None, None)
            tag = f"{cam} t{int(r.local_track_id)}" + (f" sim {own:.2f}" if own is not None else "")
            ok = best[0] in (None, cid)
            cells.append(label(_fit(crop(fr, bb, pad=0.05, size=cell), cell), tag, (255, 255, 255) if ok else (0, 0, 255)))
            row.update({f"{cam}_seen": True, f"{cam}_track": int(r.local_track_id),
                        f"{cam}_sim_own": None if own is None else round(own, 3),
                        f"{cam}_best": best[0], f"{cam}_best_sim": None if best[1] is None else round(best[1], 3)})
        rows_img.append(tile(cells, cols=len(cells), cell=cell))
        rows.append(row)
    sheet = np.vstack(rows_img) if rows_img else np.zeros((cell, cell, 3), np.uint8)
    stamp = pd.Timestamp(t, unit="s").strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"id_check_{stamp}.jpg")
    cv2.imwrite(path, sheet)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"id_check_{stamp}.csv"), index=False)
    return path, rows, sheet


def audit_text(rows: List[Dict]) -> str:
    lines = []
    for r in rows:
        cams = [k[:-5] for k in r if k.endswith("_seen") and r[k]]
        bits = []
        for cam in cams:
            own, best = r.get(f"{cam}_sim_own"), r.get(f"{cam}_best")
            bits.append(f"{cam} track {r.get(f'{cam}_track')} sim-to-own {own}" + (f" (closest card: {best})" if best and best != r["cow_id"] else ""))
        lines.append(f"{r['cow_id']}: " + ("; ".join(bits) if bits else "not seen by any camera"))
    return "\n".join(lines)

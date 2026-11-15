"""Evidence store: every fact the risk agent can cite, with provenance.

outputs/<risk_run>/evidence/
    facts.jsonl     one JSON per line: id, cow_id, kind, claim, data, images, source, time
    img/<id>.jpg    every image a tool showed the model (so a card / judge can see exactly what it saw)

Kinds
    indicator  computed behavior indicator (value, herd median, z, coverage)  - written by code
    image      an image a tool produced (camera, time, box size, daylight)    - written by tools
    visual     what the model says it sees, citing image facts                - written by the agent
    context    anything else (feed rushes, coverage gaps, data artifacts)
Facts are append-only; the final assessment must cite fact ids, which the self-check verifies.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np


class EvidenceStore:
    def __init__(self, root: str):
        self.root = root
        self.img_dir = os.path.join(root, "img")
        os.makedirs(self.img_dir, exist_ok=True)
        self.path = os.path.join(root, "facts.jsonl")
        self.facts: Dict[str, Dict] = {}
        if os.path.exists(self.path):
            for line in open(self.path):
                f = json.loads(line)
                self.facts[f["id"]] = f
        self._n = {k: sum(1 for f in self.facts.values() if f["kind"] == k) for k in ("image", "visual", "context")}

    def _next(self, prefix: str, kind: str) -> str:
        self._n[kind] = self._n.get(kind, 0) + 1
        return f"{prefix}{self._n[kind]:04d}"

    def add(self, kind: str, cow_id: Optional[str], claim: str, data: Optional[Dict] = None,
            images: Optional[List[str]] = None, source: str = "", fid: Optional[str] = None) -> str:
        fid = fid or self._next({"image": "I", "visual": "V", "context": "C"}.get(kind, "X"), kind)
        f = {"id": fid, "cow_id": cow_id, "kind": kind, "claim": claim, "data": data or {},
             "images": images or [], "source": source, "logged": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.facts[fid] = f
        with open(self.path, "a") as fh:
            fh.write(json.dumps(f, default=_js) + "\n")
        return fid

    def add_image(self, img: np.ndarray, cow_id: Optional[str], claim: str, data: Dict, source: str) -> str:
        fid = self._next("I", "image")
        path = os.path.join(self.img_dir, f"{fid}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return self.add("image", cow_id, claim, data, [os.path.relpath(path, self.root)], source, fid)

    def get(self, fid: str) -> Optional[Dict]:
        return self.facts.get(fid)

    def for_cow(self, cow_id: str, kinds=None) -> List[Dict]:
        return [f for f in self.facts.values() if f["cow_id"] == cow_id and (kinds is None or f["kind"] in kinds)]

    def image_path(self, fid: str) -> Optional[str]:
        f = self.facts.get(fid)
        return os.path.join(self.root, f["images"][0]) if f and f["images"] else None


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)

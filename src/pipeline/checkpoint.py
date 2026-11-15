"""Tracking checkpoints: a crash or Ctrl+C in a 4-hour chunk costs at most `every_sec` of footage, not the chunk.

One file, <output_dir>/tracking_checkpoint.pkl, overwritten atomically every `every_sec` seconds of footage (default
900 = 15 min) and deleted once tracking has exported its results. It holds everything tracking keeps in memory: the
records so far, the cow-identity state, the per-camera trackers, the clinical windows, the asset recorder (its CSVs
are cut back to the checkpoint on resume) and where each camera's stream stood. A 4-hour chunk's checkpoint is about
50 MB and takes well under a second to write.

A checkpoint is used only if it was made with the same inputs and settings (`fingerprint`); otherwise it is renamed
to *.stale and tracking starts from the beginning, as before.
"""
from __future__ import annotations

import os
import pickle
import time
import traceback
from typing import Any, Dict, Optional

NAME = "tracking_checkpoint.pkl"
VERSION = 1


class Checkpointer:
    def __init__(self, out_dir: str, fingerprint: Dict[str, Any], every_sec: float):
        self.path = os.path.join(out_dir, NAME)
        self.fp = {**fingerprint, "version": VERSION}
        self.every_sec = every_sec

    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "rb") as f:
                d = pickle.load(f)
        except Exception as e:  # torn or from older code: start over rather than fail
            print(f"[CHECKPOINT] unreadable ({e}); starting from the beginning", flush=True)
            os.replace(self.path, self.path + ".stale")
            return None
        if d.get("fingerprint") != self.fp:
            print("[CHECKPOINT] made with other inputs/settings; starting from the beginning", flush=True)
            os.replace(self.path, self.path + ".stale")
            return None
        return d

    def save(self, state: Dict[str, Any]) -> float:
        t = time.time()
        state["fingerprint"] = self.fp
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)  # atomic: a crash while saving keeps the previous checkpoint
        return time.time() - t

    def clear(self):
        for p in (self.path, self.path + ".tmp"):
            if os.path.exists(p):
                os.remove(p)


def log_error(out_dir: str, where: str, exc: BaseException) -> None:
    with open(os.path.join(out_dir, "step_errors.log"), "a") as f:
        f.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} {where}\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

from __future__ import annotations

from collections import defaultdict, deque


class CowMemory:
    """Short-term per-cow memory of the agent's final labels. Fills Brief.prev_label when the caller doesn't."""

    def __init__(self, maxlen: int = 30):
        self._h: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, cow_id: str, day: int, minute: int, label: str, conf: float):
        self._h[cow_id].append((day, minute, label, conf))

    def prev_label(self, cow_id: str, day: int, minute: int):
        for d, m, lab, _ in reversed(self._h[cow_id]):
            if d == day and m == minute - 1:
                return lab
        return None

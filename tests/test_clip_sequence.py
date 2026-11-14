"""ClipSequenceReader: continuous timeline across Event-named clips, gaps, shared grid."""
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from src.video.clip_sequence import ClipSequenceReader, list_clips


def _clip(path, n, fps=10, val=0):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), (val + i) % 255, np.uint8))
    w.release()


class TestClipSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.d, "sub"))
        _clip(os.path.join(cls.d, "Event20261107070000003.mp4"), 50)            # 07:00:00-07:00:05
        _clip(os.path.join(cls.d, "sub", "Event20261107070005003.mp4"), 50)     # contiguous, nested
        _clip(os.path.join(cls.d, "Event20261107070100003.mp4"), 30)            # after a 50 s hole

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def test_listing_sorted_recursive(self):
        names = [os.path.basename(c.path) for c in list_clips(self.d)]
        self.assertEqual(names, ["Event20261107070000003.mp4", "Event20261107070005003.mp4",
                                 "Event20261107070100003.mp4"])

    def test_continuous_then_gap(self):
        r = ClipSequenceReader(self.d, target_fps=2, emit_gaps=True)
        rows = [(t, f is not None, r.segment_id, r.current_clip) for _, t, f in r.frames()]
        ts = [x[0] for x in rows]
        self.assertTrue(np.allclose(np.diff(ts), 0.5))  # one grid, no jumps
        real = [x for x in rows if x[1]]
        # 07:00:00-07:00:10 contiguous (20 ticks), then 07:01:00-07:01:03 (6 ticks)
        self.assertEqual(len(real), 26)
        self.assertEqual({x[2] for x in real if x[0] < 10}, {0})  # boundary is NOT a gap
        self.assertEqual({x[2] for x in real if x[0] >= 60}, {1})  # the hole IS a gap
        self.assertEqual(sum(1 for x in rows if not x[1]), 100)   # 10 s .. 60 s of gap ticks

    def test_wall_window_and_offsets(self):
        r = ClipSequenceReader(self.d, target_fps=1, origin="2026-11-07 07:00:00",
                               wall_start="2026-11-07 07:00:03", wall_end="2026-11-07 07:00:07")
        got = [(t, r.current_clip, r.clip_offset_sec) for _, t, f in r.frames()]
        self.assertEqual([g[0] for g in got], [3, 4, 5, 6])
        self.assertEqual(got[2][1], "Event20261107070005003.mp4")
        self.assertAlmostEqual(got[2][2], 0.0)


if __name__ == "__main__":
    unittest.main()

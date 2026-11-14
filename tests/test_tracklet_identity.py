"""Tests for tracklet-level identity: uniqueness, cross-camera agreement, re-acquisition, persistence."""

import os
import tempfile
import unittest

import numpy as np

from src.tracking.local_tracker import LocalTrack, TrackState
from src.tracking.tracklet_identity import TrackletIdentityManager


class BBoxTransformer:
    """Returns the world position registered for each bbox ground point."""

    def __init__(self):
        self.pos = {}

    def pixel_to_world(self, cam, u, v, current_res=None):
        return self.pos.get((cam, round(u), round(v)))


RNG = np.random.default_rng(0)
DIM = 64
COWS = [v / np.linalg.norm(v) for v in RNG.normal(size=(6, DIM))]


def emb(i, noise=0.25):
    v = COWS[i] + noise * RNG.normal(size=DIM) / np.sqrt(DIM)
    return v / np.linalg.norm(v)


class World:
    """Places cow i of a camera at a world position and returns tracks + embeddings."""

    def __init__(self, tr):
        self.tr = tr

    def track(self, cam, tid, x, y):
        bbox = (x * 100.0, y * 100.0, x * 100.0 + 50, y * 100.0 + 80)
        u, v = (bbox[0] + bbox[2]) / 2.0, bbox[3]
        self.tr.pos[(cam, round(u), round(v))] = (x, y)
        return LocalTrack(track_id=tid, camera_id=cam, state=TrackState.CONFIRMED, bbox=bbox, detection_confidence=0.9)


def step(mgr, w, t, obs):
    """obs: list of (cam, track_id, cow_index, x, y)."""
    cams, embs = {}, {}
    for cam, tid, ci, x, y in obs:
        cams.setdefault(cam, []).append(w.track(cam, tid, x, y))
        embs[(cam, tid)] = emb(ci)
    mgr.associate_tracks(cams, embs, float(t))
    return {(cam, tr.track_id): tr.global_cow_id for cam, trs in cams.items() for tr in trs}


class TestTrackletIdentity(unittest.TestCase):
    def setUp(self):
        self.tr = BBoxTransformer()
        self.w = World(self.tr)
        self.mgr = TrackletIdentityManager(self.tr)

    def test_unique_ids_per_camera(self):
        for t in range(20):
            ids = step(self.mgr, self.w, t, [("cam1", k + 1, k, 2.0 * k, 5.0) for k in range(5)])
            vals = [v for v in ids.values() if v]
            self.assertEqual(len(vals), len(set(vals)), f"duplicate IDs at t={t}: {ids}")
        self.assertEqual(len(set(vals)), 5)

    def test_ids_stable_over_time(self):
        first = step(self.mgr, self.w, 0, [("cam1", k + 1, k, 2.0 * k, 5.0) for k in range(4)])
        for t in range(1, 30):
            ids = step(self.mgr, self.w, t, [("cam1", k + 1, k, 2.0 * k + 0.02 * t, 5.0) for k in range(4)])
            self.assertEqual(ids, first)

    def test_cross_camera_agreement(self):
        for t in range(10):
            ids = step(self.mgr, self.w, t, [("cam1", 1, 0, 3.0, 2.0), ("cam1", 2, 1, 9.0, 2.0),
                                              ("cam2", 7, 0, 3.1, 2.1), ("cam2", 8, 1, 8.9, 2.0)])
        self.assertEqual(ids[("cam1", 1)], ids[("cam2", 7)])
        self.assertEqual(ids[("cam1", 2)], ids[("cam2", 8)])
        self.assertNotEqual(ids[("cam1", 1)], ids[("cam1", 2)])

    def test_reacquire_after_track_break(self):
        for t in range(15):
            ids = step(self.mgr, self.w, t, [("cam1", 1, 0, 2.0, 5.0), ("cam1", 2, 1, 8.0, 5.0)])
        cow_a = ids[("cam1", 1)]
        # cow 0 disappears for 30 s, comes back elsewhere with a NEW ByteTrack id
        for t in range(15, 45):
            step(self.mgr, self.w, t, [("cam1", 2, 1, 8.0, 5.0)])
        ids = step(self.mgr, self.w, 45, [("cam1", 2, 1, 8.0, 5.0), ("cam1", 9, 0, 5.0, 12.0)])
        self.assertEqual(ids[("cam1", 9)], cow_a)

    def test_impossible_jump_is_rejected(self):
        for t in range(10):
            ids = step(self.mgr, self.w, t, [("cam1", 1, 0, 1.0, 1.0)])
        a = ids[("cam1", 1)]
        # 1 s later a look-alike appears 25 m away: must not be the same cow
        ids = step(self.mgr, self.w, 10, [("cam1", 1, 0, 1.0, 1.0), ("cam1", 5, 0, 1.0, 26.0)])
        self.assertNotEqual(ids[("cam1", 5)], a)

    def test_gallery_persists_across_runs(self):
        for t in range(20):
            ids = step(self.mgr, self.w, t, [("cam1", k + 1, k, 3.0 * k, 5.0) for k in range(3)])
        path = os.path.join(tempfile.mkdtemp(), "gal.npz")
        self.mgr.save_gallery(path)
        mgr2 = TrackletIdentityManager(self.tr, gallery_path=path)
        # next clip: new track ids, different order and positions
        ids2 = step(mgr2, self.w, 1000, [("cam1", 50, 2, 1.0, 20.0), ("cam1", 51, 0, 6.0, 20.0), ("cam1", 52, 1, 11.0, 20.0)])
        self.assertEqual(ids2[("cam1", 51)], ids[("cam1", 1)])
        self.assertEqual(ids2[("cam1", 52)], ids[("cam1", 2)])
        self.assertEqual(ids2[("cam1", 50)], ids[("cam1", 3)])


if __name__ == "__main__":
    unittest.main()

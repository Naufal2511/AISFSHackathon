"""Fuzz the cross-camera identity step with the inputs that show up after hours of footage: tiny / night-black
boxes (zero colour part), missing DINOv2 vectors, old-length vectors, impossible positions, a gallery of another
length. Nothing may raise, and the saved gallery must reload."""
import numpy as np

from src.tracking.tracklet_identity import TrackletIdentityManager, color_descriptor, fuse_embedding


class _T:
    def __init__(self, rng):
        self.rng = rng

    def pixel_to_world(self, cam, x, y, current_res=None):
        r = self.rng.random()
        if r < 0.2:
            return None
        return (float(self.rng.uniform(0, 12)), float(self.rng.uniform(0, 30)))


class _Track:
    def __init__(self, tid, rng):
        self.track_id = tid
        x, y = rng.uniform(0, 3000, 2)
        w, h = rng.choice([2, 5, 9, 50, 400]), rng.choice([2, 5, 9, 50, 300])
        self.bbox = (x, y, x + w, y + h)
        self.occlusion_score = float(rng.random())
        self.global_cow_id = None


def _emb(rng, frame, bbox):
    k = rng.integers(0, 6)
    dino = None if k == 0 else np.zeros(384, np.float32) if k == 1 else rng.normal(size=384).astype(np.float32)
    if k == 5:  # an old 384-only or a colour-only vector reaching the manager directly
        return rng.normal(size=int(rng.choice([384, 36]))).astype(np.float32)
    return fuse_embedding(dino, color_descriptor(frame, bbox))


def test_identity_step_never_raises(tmp_path):
    rng = np.random.default_rng(0)
    gal = tmp_path / "g.npz"
    np.savez(gal, cow_ids=np.array(["COW_01", "COW_02"]), protos=np.ones((2, 384), np.float32),
             n_updates=np.array([3, 3]))
    m = TrackletIdentityManager(_T(rng), num_cows=10, gallery_path=str(gal))
    black = np.zeros((3000, 4000, 3), np.uint8)
    noisy = rng.integers(0, 255, (3000, 4000, 3), dtype=np.uint8)
    for step in range(400):
        frame = black if step % 3 == 0 else noisy
        cams = {}
        embs = {}
        for cam in ("cam1", "cam2", "cam3"):
            ts = [_Track(int(rng.integers(0, 15)), rng) for _ in range(int(rng.integers(0, 12)))]
            ts = list({t.track_id: t for t in ts}.values())
            cams[cam] = ts
            for t in ts:
                e = _emb(rng, frame, t.bbox)
                if e is not None:
                    embs[(cam, t.track_id)] = e
        m.associate_tracks(cams, embs, float(step), {"cam1": (4000, 3000)})
    out = tmp_path / "out.npz"
    m.save_gallery(str(out))
    TrackletIdentityManager(_T(rng), num_cows=10, gallery_path=str(out))


def test_fused_length_is_fixed():
    rng = np.random.default_rng(1)
    for color in (None, np.zeros(36, np.float32), rng.random(36).astype(np.float32)):
        assert fuse_embedding(rng.normal(size=384), color).shape == (420,)
    assert fuse_embedding(None, rng.random(36)) is None
    assert fuse_embedding(np.zeros(384), rng.random(36)) is None

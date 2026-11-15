"""3-laptop split (run_share.py) and the stitcher (stitch_shares.py): no videos, no models."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_share
import stitch_shares
from pipeline import cameras as camlib
from pipeline.orchestrator import minute_activity_with_wall_time


def _clip(cam, start):
    return camlib.ClipInfo(path=f"/v/{cam}/{start}", name=f"Event{start}.Avi", camera=cam, rule="suffix",
                           start=pd.Timestamp(start).isoformat())


def test_chunks_cover_share_on_minute_edges():
    ch = run_share.chunks_of({"start": "2026-11-07 00:04", "end": "2026-11-07 08:40"}, chunk_hours=4, parallel=2)
    assert ch[0][0] == "2026-11-07 00:04" and ch[-1][1] == "2026-11-07 08:40"
    assert all(a[1] == b[0] for a, b in zip(ch, ch[1:]))           # no gap, no overlap
    assert len(ch) == 3                                             # 8.6 h / 4 h -> 3 chunks (>= parallel)


def test_intervals_follow_clip_spacing():
    clips = [_clip("cam1", "2026-11-07 10:00:00"), _clip("cam1", "2026-11-07 10:05:00"), _clip("cam1", "2026-11-07 11:00:00")]
    iv = run_share._intervals(clips)
    lens = sorted((b - a).total_seconds() for a, b in iv)
    assert lens == [300.0, 300.0, 900.0]                            # gap of 55 min capped at 15 min; last = median


def test_clips_in_window_keeps_the_clip_running_into_the_start():
    clips = [_clip("cam1", "2026-11-07 09:50:00"), _clip("cam1", "2026-11-07 10:01:00"), _clip("cam1", "2026-11-07 11:00:00")]
    got = [c.start for c in camlib.clips_in_window(clips, "2026-11-07 10:00", "2026-11-07 11:00")]
    assert got == ["2026-11-07T09:50:00", "2026-11-07T10:01:00"]


def _chunk(tmp, name, start, end, protos):
    d = tmp / "naufal_training" / name
    mem = d / "stage1" / "level1" / "agent" / "memory"
    mem.mkdir(parents=True)
    np.savez(mem / "prototypes.npz", **protos)
    return {"name": name, "who": "naufal_training", "dir": d, "start": pd.Timestamp(start), "end": pd.Timestamp(end),
            "run_start": pd.Timestamp(start)}


def test_align_ids_matches_by_appearance_and_flags_new(tmp_path):
    rng = np.random.default_rng(0)
    a, b, c = (rng.normal(size=16) for _ in range(3))
    c1 = _chunk(tmp_path, "c01", "2026-11-07 10:00", "2026-11-07 11:00", {"COW_01": a, "COW_02": b})
    # chunk 2 numbered the same two cows the other way round, and saw a third one
    c2 = _chunk(tmp_path, "c02", "2026-11-07 11:00", "2026-11-07 12:00",
                {"COW_01": b + 0.05 * rng.normal(size=16), "COW_02": a + 0.05 * rng.normal(size=16), "COW_03": c})
    m = stitch_shares.align_ids([c1, c2], min_sim=0.55, min_margin=0.05, overrides={})
    g = dict(zip(m[m.chunk == "c02"].local_id, m[m.chunk == "c02"].global_id))
    assert g["COW_01"] == "COW_02" and g["COW_02"] == "COW_01"
    assert g["COW_03"] == "COW_03" and m.set_index(["chunk", "local_id"]).loc[("c02", "COW_03"), "status"] == "new"
    o = stitch_shares.align_ids([c1, c2], 0.55, 0.05, {"c02": {"COW_03": "COW_01"}, "c01": {}})
    assert o[(o.chunk == "c02") & (o.local_id == "COW_03")].global_id.item() == "COW_01"


def test_overlap_boxes_beat_appearance(tmp_path):
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=16), rng.normal(size=16)
    c1 = _chunk(tmp_path, "c01", "2026-11-07 10:00", "2026-11-07 11:00", {"COW_01": a, "COW_02": b})
    # chunk 2 swapped the numbering AND its prototypes look like the wrong cows; the shared lead-in boxes decide
    c2 = _chunk(tmp_path, "c02", "2026-11-07 11:00", "2026-11-07 12:00", {"COW_01": a, "COW_02": b})
    c2["run_start"] = pd.Timestamp("2026-11-07 10:58")
    secs = pd.date_range("2026-11-07 10:58", periods=30, freq="s").strftime("%Y-%m-%dT%H:%M:%S")
    def obs(ch, ids):
        p = ch["dir"] / "stage1" / "level1" / "contract"
        p.mkdir(parents=True, exist_ok=True)
        rows = [{"wall_time": t, "camera": "cam1", "cow_id": cid, "x1": x, "y1": 0, "x2": x + 100, "y2": 100}
                for t in secs for cid, x in zip(ids, (0, 500))]
        pd.DataFrame(rows).to_csv(p / "observations.csv", index=False)
    obs(c1, ["COW_01", "COW_02"])
    obs(c2, ["COW_02", "COW_01"])                    # the left cow is COW_01 in c01 and COW_02 in c02
    m = stitch_shares.align_ids([c1, c2], 0.55, 0.05, {})
    g = m[m.chunk == "c02"].set_index("local_id")
    assert g.loc["COW_02", "global_id"] == "COW_01" and g.loc["COW_01", "global_id"] == "COW_02"
    assert g.status.str.startswith("overlap").all()



def test_overlap_split_across_cameras_is_kept_and_flagged(tmp_path):
    """Seen on c01/c02: each chunk links cam1 and cam3 boxes into cows on its own. Here c02 calls the cam1 box and
    the cam3 box one cow; c01 calls them two. The boxes still match perfectly per camera, so the match is taken
    from the overlap (flagged), not from near-tied appearance; equal votes are broken by appearance."""
    rng = np.random.default_rng(2)
    a, b, c = (rng.normal(size=16) for _ in range(3))
    c1 = _chunk(tmp_path, "c01", "2026-11-07 10:00", "2026-11-07 11:00", {"COW_01": a, "COW_02": b})
    c2 = _chunk(tmp_path, "c02", "2026-11-07 11:00", "2026-11-07 12:00",
                {"COW_01": b + 0.05 * rng.normal(size=16), "COW_02": a + 0.05 * rng.normal(size=16), "COW_03": c})
    c2["run_start"] = pd.Timestamp("2026-11-07 10:58")
    secs = pd.date_range("2026-11-07 10:58", periods=30, freq="s").strftime("%Y-%m-%dT%H:%M:%S")
    def obs(ch, boxes):                              # boxes: [(camera, cow_id, x)]
        p = ch["dir"] / "stage1" / "level1" / "contract"
        p.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"wall_time": t, "camera": cam, "cow_id": cid, "x1": x, "y1": 0, "x2": x + 100, "y2": 100}
                      for t in secs for cam, cid, x in boxes]).to_csv(p / "observations.csv", index=False)
    obs(c1, [("cam1", "COW_01", 0), ("cam3", "COW_02", 300)])
    obs(c2, [("cam1", "COW_01", 0), ("cam3", "COW_01", 300)])     # c02: both boxes are one cow
    m = stitch_shares.align_ids([c1, c2], 0.55, 0.05, {})
    g = m[m.chunk == "c02"].set_index("local_id")
    assert g.loc["COW_01", "status"].startswith("overlap_split")    # kept from the overlap, flagged
    assert g.loc["COW_01", "global_id"] == "COW_02"                 # 30 vs 30 votes: appearance breaks the tie
    assert g.loc["COW_01", "overlap_detail"] == "cam1 COW_01 30; cam3 COW_02 30"
    assert g.loc["COW_02", "global_id"] == "COW_01"                 # the rest by appearance
    assert g.loc["COW_03", "status"] == "new"
    assert m.global_id.nunique() == 3

def _mins(rows):
    return pd.DataFrame([{"cow_id": c, "minute_start": t, "behavior": beh, "confidence": 0.9, "reason": r, "source": "model_estimate",
                          "correction": "confirmed", "tier": 2 if r.startswith("agent") else 1, "flags": "", "cameras_used": "['front']"}
                         for c, t, beh, r in rows])


def test_activity_meets_the_brief():
    from pipeline import deliverables as dv
    m = _mins([("COW_01", "2026-11-07 16:00", "ruminating_lying", "agent_accept"),
               ("COW_01", "2026-11-07 16:01", "ruminating_lying", "stable_carry"),
               ("COW_01", "2026-11-07 16:02", "ruminating_lying", "keep_tier1"),
               ("COW_02", "2026-11-08 09:00", "social_interaction", "unresolved_fallback_tier1")])
    g = dv.full_grid(m, cows=["COW_01", "COW_02", "COW_09"])
    assert len(g) == 3 * 2 * 1440                                   # every cow, every minute of both days
    b = dv.bouts(g)
    rum = b[(b.cow_id == "COW_01") & (b.behavior == "ruminating (lying)")].iloc[0]   # official class names
    assert (rum.start_time, rum.end_time, rum.duration_min) == ("2026-11-07 16:00", "2026-11-07 16:03", 3)
    assert rum.basis == "estimated" and rum.observed_share   # 1 verified, 1 carried, 1 observed: tie -> least certain == round(2 / 3, 3) and rum.n_agent_minutes == 1
    assert b[b.cow_id == "COW_09"].behavior.unique().tolist() == ["other / not visible"]
    assert set(b[b.cow_id == "COW_09"].basis) == {"not_visible"}
    assert b.groupby("cow_id").duration_min.sum().tolist() == [2880, 2880, 2880]
    assert dv.basis_of({"reason": "unresolved_fallback_tier1", "behavior": "social_interaction"}) == "unverified"
    tb = dv.time_budgets(g)
    one = tb[(tb.cow_id == "COW_01") & (tb.date == "all") & (tb.behavior == "ruminating (lying)")].iloc[0]
    assert one.minutes == 3 and one.share_of_visible_time == 1.0


def test_bouts_join_across_chunk_edge():
    from pipeline import deliverables as dv
    m = _mins([("COW_01", f"2026-11-07 10:{k:02d}", "lying", "keep_tier1") for k in (0, 1, 2, 4)])
    b = dv.bouts(dv.full_grid(m))
    lying = b[b.behavior == "lying"]
    assert lying[["start_minute", "end_minute", "duration_min"]].values.tolist() == [[600, 602, 3], [604, 604, 1]]


def test_minute_activity_gets_wall_time(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"clock_origin": "2026-11-07T00:00:00"}))
    pd.DataFrame({"cow_id": ["COW_01"], "day": [2], "minute": [960], "behavior": ["lying"], "confidence": [0.9]}).to_csv(
        tmp_path / "activity_minute.csv", index=False)
    n = minute_activity_with_wall_time(tmp_path / "activity_minute.csv", tmp_path / "meta.json", tmp_path / "out" / "activity.csv")
    got = pd.read_csv(tmp_path / "out" / "activity.csv")
    assert n == 1 and got.minute_start.item() == "2026-11-08T16:00:00"
    from src.risk.data import load_activity                         # Level 2 reads it on the right clock
    la = load_activity(tmp_path / "out" / "activity.csv")
    assert la.minute.item() == int((pd.Timestamp("2026-11-08 16:00") - pd.Timestamp("1970-01-01")).total_seconds())


def test_missing_clips_are_found_per_window():
    plan = {"shares": {"a": {"clips": ["Event20261107095500006.Avi", "Event20261107100000006.Avi", "Event20261107100500017.Avi"]},
                       "b": {"clips": ["Event20261107120000003.Avi"]}}}
    have = {"event20261107100000006.avi"}
    # 10:00-10:10 needs the clip running into 10:00 (09:55), the 10:00 one and cam2's 10:05; not the 12:00 one
    assert run_share.missing_for_window(plan, have, "2026-11-07 10:00", "2026-11-07 10:10") == \
        ["Event20261107095500006.Avi", "Event20261107100500017.Avi"]
    assert run_share.missing_for_window(plan, have | {"event20261107095500006.avi", "event20261107100500017.avi"},
                                        "2026-11-07 10:00", "2026-11-07 10:10") == []


def test_device_override(monkeypatch):
    from src.device import pick_device
    monkeypatch.setenv("PEN6_DEVICE", "cpu")
    assert pick_device() == "cpu"

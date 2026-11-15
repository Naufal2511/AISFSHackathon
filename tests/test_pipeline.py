"""run_pipeline.py: camera assignment, input layout, resume bookkeeping (no models, no real video)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from pipeline import cameras as camlib
from pipeline import orchestrator as orch

CFG = {"folder_pattern": r"^cam(?:era)?[\s_-]*0?([1-3])$",
       "suffix": {"006": "cam1", "017": "cam2", "003": "cam3"},
       "expected_resolution": {"cam1": [4000, 3000], "cam2": [2592, 1944], "cam3": [4000, 3000]}}


def touch(p: Path, data: bytes = b"x" * 10) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


class CameraAssignment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_flat_folder_uses_suffix(self):
        for n in ("Event20261107160019006.Avi", "Event20261107160029017.Avi", "Event20261107160008003.Avi",
                  "Event20261107160519006.Avi", "notes.txt", "clip.avi"):
            touch(self.root / n)
        clips = camlib.assign_cameras(self.root, CFG)
        got = {c.name: (c.camera, c.rule) for c in clips}
        self.assertEqual(got["Event20261107160019006.Avi"], ("cam1", "suffix"))
        self.assertEqual(got["Event20261107160029017.Avi"], ("cam2", "suffix"))
        self.assertEqual(got["Event20261107160008003.Avi"], ("cam3", "suffix"))
        self.assertEqual(len(clips), 4)                       # non-Event files ignored
        self.assertEqual(camlib.summary(clips)["cam1"]["clips"], 2)

    def test_dvr_folders_win_and_unknown_suffix_ok_there(self):
        touch(self.root / "Day 1" / "Cam 02" / "Event20261107071945999.Avi")   # suffix unknown, folder decides
        touch(self.root / "Day 1" / "cam3" / "Event20261107071935003.Avi")
        clips = {c.name: c for c in camlib.assign_cameras(self.root, CFG)}
        self.assertEqual((clips["Event20261107071945999.Avi"].camera, clips["Event20261107071945999.Avi"].rule),
                         ("cam2", "folder"))
        self.assertEqual(clips["Event20261107071935003.Avi"].start, "2026-11-07T07:19:35")

    def test_conflict_and_unknown_stop(self):
        touch(self.root / "Cam 01" / "Event20261107160008003.Avi")             # folder cam1, suffix cam3
        with self.assertRaises(camlib.CameraError) as e:
            camlib.assign_cameras(self.root, CFG)
        self.assertIn("disagree", str(e.exception))
        (self.root / "Cam 01" / "Event20261107160008003.Avi").unlink()
        touch(self.root / "Event20261107160008999.Avi")
        with self.assertRaises(camlib.CameraError) as e:
            camlib.assign_cameras(self.root, CFG)
        self.assertIn("no camera", str(e.exception))

    def test_generic_folder_names_are_not_cameras(self):
        touch(self.root / "back" / "Event20261107152517006.Avi")   # 'back' is ambiguous (2 front cams): suffix decides
        (c,) = camlib.assign_cameras(self.root, CFG)
        self.assertEqual((c.camera, c.rule), ("cam1", "suffix"))

    def test_resolution_check(self):
        touch(self.root / "Event20261107160019006.Avi")
        touch(self.root / "Event20261107160029017.Avi")
        clips = camlib.assign_cameras(self.root, CFG)
        sizes = {"Event20261107160019006.Avi": (4000, 3000), "Event20261107160029017.Avi": (2592, 1944)}
        ok = camlib.check_resolutions(clips, CFG["expected_resolution"], probe=lambda p: sizes[Path(p).name])
        self.assertEqual(ok, {"cam1": [4000, 3000], "cam2": [2592, 1944]})
        sizes["Event20261107160029017.Avi"] = (4000, 3000)
        with self.assertRaises(camlib.CameraError):
            camlib.check_resolutions(clips, CFG["expected_resolution"], probe=lambda p: sizes[Path(p).name])

    def test_link_inputs_layout_and_relink(self):
        src = self.root / "src"
        a = touch(src / "Event20261107160019006.Avi", b"a" * 100)
        touch(src / "Event20261107160029017.Avi", b"b" * 100)
        clips = camlib.assign_cameras(src, CFG)
        dest = self.root / "run" / "input"
        st = camlib.link_inputs(clips, dest)
        self.assertEqual(st["linked"] + st["copied"], 2)
        self.assertEqual((dest / "cam1" / a.name).read_bytes(), b"a" * 100)
        self.assertTrue((dest / "cam2" / "Event20261107160029017.Avi").exists())
        st = camlib.link_inputs(clips, dest)                       # idempotent
        self.assertEqual(st["kept"], 2)
        touch(dest / "cam1" / "Event20261107150000006.Avi")        # stale file from an older assignment
        st = camlib.link_inputs(clips, dest)
        self.assertEqual(st["removed"], 1)
        self.assertFalse((dest / "cam1" / "Event20261107150000006.Avi").exists())


class Resume(unittest.TestCase):
    def test_state_roundtrip_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            vids = Path(tmp) / "videos"
            touch(vids / "Event20261107160019006.Avi")
            out_root = Path(tmp) / "runs"
            rc = orch.main(["--videos", str(vids), "--out-root", str(out_root), "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertEqual(orch._default_name(str(vids), {"cameras": CFG}), "run_20261107_1600")
            run = orch.Run(out_root / "r1")
            run.save_state({"videos": str(vids.resolve()), "steps": {s: {"status": "ok"} for s in orch.STEPS}})
            # every step done -> nothing to do, returns 0 without preflight
            self.assertEqual(orch.main(["--name", "r1", "--out-root", str(out_root)]), 0)
            self.assertEqual(json.loads(run.state_file.read_text())["videos"], str(vids.resolve()))
            # same name, different videos -> refuses instead of mixing runs
            other = Path(tmp) / "other"
            touch(other / "Event20261107160019006.Avi")
            with self.assertRaises(SystemExit):
                orch.main(["--videos", str(other), "--name", "r1", "--out-root", str(out_root)])


if __name__ == "__main__":
    unittest.main()

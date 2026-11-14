#!/usr/bin/env python3
"""Bridge between data/annotation (YOLO) and a LOCAL Label Studio instance.

  # 1. build tasks (images served from local disk, pre-labels as editable predictions)
  python scripts/labelstudio_bridge.py make-tasks --data data/annotation

  # 2. after labelling: Export -> "JSON" in Label Studio, then write YOLO labels back
  python scripts/labelstudio_bridge.py to-yolo --export ~/Downloads/project-1-at-....json --data data/annotation

Only SUBMITTED annotations are written back; frames you have not reviewed keep their
pre-label file but are listed as unreviewed (and excluded from training with --only-reviewed).
"""
import argparse
import glob
import json
import os
import sys
from urllib.parse import parse_qs, unquote, urlparse

import cv2

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" strokeWidth="3">
    <Label value="cow" background="#FF4136" hotkey="1"/>
  </RectangleLabels>
</View>
"""


def make_tasks(a):
    root = os.path.abspath(a.doc_root)
    already = set()
    for prev in a.exclude or []:  # task files already imported into Label Studio
        already |= {t["data"]["image"] for t in json.load(open(prev))}
    tasks = []
    for img in sorted(glob.glob(os.path.join(a.data, "images", "*", "*.jpg"))):
        rel = os.path.relpath(os.path.abspath(img), root)
        if f"/data/local-files/?d={rel}" in already:
            continue
        split = os.path.basename(os.path.dirname(img))
        lbl = os.path.join(a.data, "labels", split, os.path.splitext(os.path.basename(img))[0] + ".txt")
        h, w = cv2.imread(img).shape[:2]
        result = []
        if os.path.exists(lbl):
            for i, line in enumerate(open(lbl)):
                p = line.split()
                if len(p) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, p)
                result.append({
                    "id": f"p{i}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                    "original_width": w, "original_height": h, "image_rotation": 0,
                    "value": {"x": 100 * (cx - bw / 2), "y": 100 * (cy - bh / 2), "width": 100 * bw,
                              "height": 100 * bh, "rotation": 0, "rectanglelabels": ["cow"]},
                })
        tasks.append({"data": {"image": f"/data/local-files/?d={rel}", "split": split,
                               "file": os.path.basename(img)},
                      "predictions": [{"model_version": "prelabel_yolo26_ensemble", "score": 0.5, "result": result}]})
    out = os.path.join(a.data, a.tasks_name)
    with open(out, "w") as f:
        json.dump(tasks, f)
    with open(os.path.join(a.data, "labelstudio_config.xml"), "w") as f:
        f.write(LABEL_CONFIG)
    print(f"wrote {len(tasks)} tasks -> {out}")
    print(f"label config -> {os.path.join(a.data, 'labelstudio_config.xml')}")
    print(f"local-files document root must be: {root}")


def _image_path(task, doc_root):
    url = task["data"]["image"]
    q = parse_qs(urlparse(url).query)
    rel = unquote(q["d"][0]) if "d" in q else unquote(url)
    return os.path.join(doc_root, rel)


def to_yolo(a):
    tasks = json.load(open(a.export))
    done, empty, skipped = 0, 0, 0
    reviewed = []
    for t in tasks:
        anns = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
        if not anns:
            skipped += 1
            continue
        ann = max(anns, key=lambda x: x.get("updated_at") or x.get("created_at") or "")
        img = _image_path(t, os.path.abspath(a.doc_root))
        split = os.path.basename(os.path.dirname(img))
        lbl = os.path.join(a.data, "labels", split, os.path.splitext(os.path.basename(img))[0] + ".txt")
        lines = []
        for r in ann.get("result", []):
            if r.get("type") != "rectanglelabels":
                continue
            v = r["value"]
            x, y, w, h = v["x"] / 100, v["y"] / 100, v["width"] / 100, v["height"] / 100
            x, y = max(0.0, x), max(0.0, y)
            w, h = min(1.0 - x, w), min(1.0 - y, h)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"0 {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
        os.makedirs(os.path.dirname(lbl), exist_ok=True)
        with open(lbl, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        reviewed.append(os.path.relpath(img, os.path.abspath(a.data)))
        done += 1
        empty += not lines
    with open(os.path.join(a.data, "reviewed.txt"), "w") as f:
        f.write("\n".join(sorted(reviewed)) + "\n")
    print(f"wrote {done} reviewed label files ({empty} with no cows); {skipped} tasks not yet annotated")
    print(f"reviewed list -> {os.path.join(a.data, 'reviewed.txt')}")
    if a.only_reviewed:
        _write_reviewed_dataset(a, reviewed)


def _write_reviewed_dataset(a, reviewed):
    """dataset_reviewed.yaml that trains/validates on reviewed frames only (via image list files)."""
    data = os.path.abspath(a.data)
    for split in ("train", "val"):
        with open(os.path.join(data, f"{split}_reviewed.txt"), "w") as f:
            for r in reviewed:
                if r.startswith(f"images/{split}/"):
                    f.write(os.path.join(data, r) + "\n")
    with open(os.path.join(data, "dataset_reviewed.yaml"), "w") as f:
        f.write(f"path: {data}\ntrain: train_reviewed.txt\nval: val_reviewed.txt\nnames:\n  0: cow\n")
    print(f"training set of reviewed frames only -> {os.path.join(data, 'dataset_reviewed.yaml')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-tasks")
    m.add_argument("--data", default="data/annotation")
    m.add_argument("--doc-root", default=".", help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (project root)")
    m.add_argument("--exclude", nargs="*", help="task JSON(s) already imported; their images are skipped")
    m.add_argument("--tasks-name", default="labelstudio_tasks.json", help="output file name inside --data")
    t = sub.add_parser("to-yolo")
    t.add_argument("--export", required=True, help="Label Studio JSON export")
    t.add_argument("--data", default="data/annotation")
    t.add_argument("--doc-root", default=".")
    t.add_argument("--only-reviewed", action="store_true", help="also write dataset_reviewed.yaml")
    a = ap.parse_args()
    make_tasks(a) if a.cmd == "make-tasks" else to_yolo(a)


if __name__ == "__main__":
    main()

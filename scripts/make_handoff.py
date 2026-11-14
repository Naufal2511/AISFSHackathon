#!/usr/bin/env python3
"""Pack both annotation datasets so a teammate can continue in their own LOCAL Label Studio.

  python scripts/make_handoff.py --out handoff/pen6_annotation --zip

Output (paths inside mirror the repo, so exports from the teammate convert here unchanged):
  pen6_annotation/
    HANDOFF.md                      step-by-step for the teammate
    start_labelstudio.sh            starts Label Studio with this folder as local-files root
    detection_tasks.json            all cow-box frames: reviewed ones carry your boxes as
                                    finished annotations, new ones carry editable pre-labels
    detection_config.xml
    behavior_tasks.json             all per-cow 5 s clips (+ any behavior labels already made)
    behavior_config.xml
    data/annotation/{images,labels,manifest.csv,...}
    data/behavior_annotation/{clips,context,crops,manifest.csv}

Finished annotations are read from the Label Studio database (latest submitted per task), so the
package holds exactly what the Label Studio UI shows, also boxes edited after the last export.
"""
import argparse
import glob
import json
import os
import shutil
import sqlite3
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from labelstudio_bridge import LABEL_CONFIG  # noqa: E402

DEFAULT_DB = os.path.expanduser("~/Library/Application Support/label-studio/label_studio.sqlite3")


def finished_from_db(db, project_title):
    """{task image/video url: result list} for the latest non-cancelled annotation per task."""
    if not db or not os.path.exists(db):
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    pid = con.execute("select id from project where title=?", (project_title,)).fetchone()
    if not pid:
        print(f"[warn] no Label Studio project titled {project_title!r}; no finished annotations taken")
        return {}
    rows = con.execute("""select t.data, c.result, c.updated_at from task t join task_completion c
                          on c.task_id=t.id where t.project_id=? and c.was_cancelled=0
                          order by c.updated_at""", (pid[0],)).fetchall()
    out = {}
    for data, result, _ in rows:  # ordered by time -> last write wins
        d = json.loads(data)
        key = d.get("image") or d.get("video")
        out[key] = json.loads(result)
    return out


def detection_tasks(repo, done):
    import cv2
    tasks, n_done = [], 0
    for img in sorted(glob.glob(os.path.join(repo, "data/annotation/images/*/*.jpg"))):
        rel = os.path.relpath(img, repo)
        url = f"/data/local-files/?d={rel}"
        split = rel.split("/")[-2]
        task = {"data": {"image": url, "split": split, "file": os.path.basename(img)}}
        if url in done:
            task["annotations"] = [{"result": done[url]}]
            n_done += 1
        else:
            h, w = cv2.imread(img).shape[:2]
            lbl = os.path.join(repo, "data/annotation/labels", split, os.path.splitext(os.path.basename(img))[0] + ".txt")
            res = []
            for i, line in enumerate(open(lbl) if os.path.exists(lbl) else []):
                p = line.split()
                if len(p) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, p)
                res.append({"id": f"p{i}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                            "original_width": w, "original_height": h, "image_rotation": 0,
                            "value": {"x": 100 * (cx - bw / 2), "y": 100 * (cy - bh / 2), "width": 100 * bw,
                                      "height": 100 * bh, "rotation": 0, "rectanglelabels": ["cow"]}})
            task["predictions"] = [{"model_version": "prelabel_yolo26_ensemble", "score": 0.5, "result": res}]
        tasks.append(task)
    return tasks, n_done


def behavior_tasks(repo, done):
    tasks, seen, n_done = [], set(), 0
    for f in sorted(glob.glob(os.path.join(repo, "data/behavior_annotation/labelstudio_tasks_seed*.json"))):
        for t in json.load(open(f)):
            if t["data"]["task_id"] in seen:
                continue
            seen.add(t["data"]["task_id"])
            if t["data"]["video"] in done:
                t["annotations"] = [{"result": done[t["data"]["video"]]}]
                t.pop("predictions", None)
                n_done += 1
            tasks.append(t)
    return tasks, n_done


def copy_tree(src, dst, patterns):
    n = 0
    for pat in patterns:
        for f in glob.glob(os.path.join(src, pat), recursive=True):
            if os.path.isdir(f):
                continue
            to = os.path.join(dst, os.path.relpath(f, src))
            os.makedirs(os.path.dirname(to), exist_ok=True)
            shutil.copy2(f, to)
            n += 1
    return n


START_SH = """#!/usr/bin/env bash
# Start a LOCAL Label Studio whose local-files root is THIS folder (data never leaves the machine).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$HERE"
LS="$(command -v label-studio || true)"
[ -x "$HERE/.venv/bin/label-studio" ] && LS="$HERE/.venv/bin/label-studio"
if [ -z "$LS" ]; then
  echo "Label Studio not found. Install once:  python3 -m venv \\"$HERE/.venv\\" && \\"$HERE/.venv/bin/pip\\" install label-studio"
  exit 1
fi
echo "Label Studio -> http://localhost:8080   (local-files root: $HERE)"
exec "$LS" start --port 8080 "$@"
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="handoff/pen6_annotation")
    ap.add_argument("--ls-db", default=DEFAULT_DB, help="Label Studio sqlite DB with your finished work")
    ap.add_argument("--detection-project", default="pen6 bb data")
    ap.add_argument("--behavior-project", default="Pen6 behavior")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    repo, out = os.path.abspath(a.repo), os.path.abspath(a.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    det, det_done = detection_tasks(repo, finished_from_db(a.ls_db, a.detection_project))
    beh, beh_done = behavior_tasks(repo, finished_from_db(a.ls_db, a.behavior_project))
    json.dump(det, open(os.path.join(out, "detection_tasks.json"), "w"))
    json.dump(beh, open(os.path.join(out, "behavior_tasks.json"), "w"))
    open(os.path.join(out, "detection_config.xml"), "w").write(LABEL_CONFIG)
    bcfg = os.path.join(repo, "data/behavior_annotation/labelstudio_config.xml")
    if os.path.exists(bcfg):
        shutil.copy2(bcfg, os.path.join(out, "behavior_config.xml"))

    n1 = copy_tree(os.path.join(repo, "data/annotation"), os.path.join(out, "data/annotation"),
                   ["images/**/*.jpg", "labels/**/*.txt", "manifest.csv", "classes.txt", "reviewed.txt"])
    n2 = copy_tree(os.path.join(repo, "data/behavior_annotation"), os.path.join(out, "data/behavior_annotation"),
                   ["clips/*.mp4", "context/*.jpg", "crops/**/*.jpg", "manifest.csv"])
    with open(os.path.join(out, "start_labelstudio.sh"), "w") as f:
        f.write(START_SH)
    os.chmod(os.path.join(out, "start_labelstudio.sh"), 0o755)
    guide = os.path.join(repo, "docs/HANDOFF_TEAMMATE.md")
    if os.path.exists(guide):
        shutil.copy2(guide, os.path.join(out, "HANDOFF.md"))

    print(f"detection: {len(det)} tasks ({det_done} already labelled, {len(det) - det_done} to do)")
    print(f"behavior : {len(beh)} tasks ({beh_done} already labelled, {len(beh) - beh_done} to do)")
    print(f"files    : {n1} detection, {n2} behavior -> {out}")
    if a.zip:
        z = out.rstrip("/") + ".zip"
        if os.path.exists(z):
            os.remove(z)
        subprocess.run(["ditto", "-c", "-k", "--keepParent", out, z], check=True)
        print(f"zip      : {z}  ({os.path.getsize(z) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()

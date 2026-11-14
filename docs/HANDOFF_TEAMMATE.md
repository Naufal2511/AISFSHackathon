# Pen 6 annotation hand-off

Two datasets, two Label Studio projects, all on your own machine.
**Event data: keep this folder local. No cloud tools (Roboflow, CVAT cloud, Google Drive sync of exports).**

| Project | What you do | Tasks file | Config |
|---|---|---|---|
| **Pen6 cows** (detection) | fix a box around every cow | `detection_tasks.json` | `detection_config.xml` |
| **Pen6 behavior** | watch a 5 s clip of one cow, pick its behavior | `behavior_tasks.json` | `behavior_config.xml` |

Frames that are already labelled arrive as **finished annotations** (they show as completed;
you can open and fix them). New frames arrive with **pre-labels** you correct.

## 1. Install and start (once, ~5 min)

```bash
cd pen6_annotation
python3 -m venv .venv && .venv/bin/pip install label-studio
./start_labelstudio.sh          # opens http://localhost:8080
```

Windows (PowerShell / cmd), from this folder:

```
py -m venv .venv
.venv\Scripts\pip install label-studio
start_labelstudio.bat
```

Unzip it anywhere: every task points to files *relative to this folder*, and the start script
makes this folder the file root. Always start Label Studio with the script, from wherever the
folder now lives, or images and clips will not load. Sign up in the browser (the account only
exists on your machine).

## 2. Create the two projects (once, ~3 min each)

For each project:
1. **Create Project**, name it (`Pen6 cows` / `Pen6 behavior`).
2. **Labeling Setup → Custom template → Code**: paste the whole config file → **Save**.
3. **Settings → Cloud Storage → Add Source Storage** → type **Local files**, absolute path =
   - cows: `<this folder>/data/annotation`
   - behavior: `<this folder>/data/behavior_annotation`
   → **Add Storage**. Do **not** press *Sync*.
4. **Settings → Annotation** → turn on **Use predictions to prelabel tasks**.
5. **Import** → upload the tasks file.

If images show as broken: the storage path in step 3 is wrong, or Label Studio was not started
with `start_labelstudio.sh`.

## 3. Label

Click **Label All Tasks**: it serves the unfinished ones. **Submit** = Ctrl/Cmd + Enter.

### Cows (detection)
- Press `1`, drag a box. Resize by the corners. Delete: click the box, then Backspace.
- **Box every cow you can see**, any pen. Include cows cut off by the frame edge (box to the edge)
  and partly hidden cows with about 30% or more visible (box the visible part).
- Tight boxes: body + head + legs, not the tail tip. No boxes on people, dogs, birds, reflections.
- Remove duplicate pre-boxes and split boxes that cover two cows. Empty frame: just Submit.
- Zoom: scroll with Ctrl/Cmd held. Night (IR) frames count the same.

### Behavior
- Left: zoomed clip following the cow (yellow box). Right: where it is in the pen.
- Play (space), press the hotkey, Submit:

| key | behavior | key | behavior |
|---|---|---|---|
| 1 | lying | 6 | ruminating (lying) |
| 2 | standing | 7 | ruminating (standing) |
| 3 | walking | 8 | grooming |
| 4 | feeding | 9 | social interaction |
| 5 | drinking | 0 | other / not visible |

- Pick what fills **most of the 5 s**.
- *ruminating*: only when you actually see rhythmic chewing; otherwise lying / standing.
- *feeding*: head in or over the bunk. Waiting in the queue behind it = standing.
- *social interaction*: head butting, grooming another cow, mounting, pushing.
- *grooming*: licking or scratching itself.
- Flags: **bad track** if the yellow box jumps to another cow, covers two cows or is not a cow
  (dropped). **unsure** if you cannot decide (dropped). **mostly occluded** is kept.
- Some tasks come pre-selected (feeding / drinking / walking guesses). Check them, don't trust them.

## 4. Send the work back

In each project: **Export → JSON** (not "JSON-MIN"). Send the two `.json` files back.
They contain only labels and file paths, no images. Partial exports are fine; the latest one wins.

On the repo side they are converted with:

```bash
.venv/bin/python scripts/labelstudio_bridge.py to-yolo --export <cows-export>.json --only-reviewed
.venv/bin/python scripts/build_behavior_tasks.py to-cls --export <behavior-export>.json
```

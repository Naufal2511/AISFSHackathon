# Cow detector annotation guide (100–200 frames, ~2–3 h)

## 1. Generate frames + pre-labels (~15–20 min for 200)

```bash
python scripts/sample_annotation_frames.py --n 200 --out data/annotation
```

Frames are stratified by camera and day/night, whole clips go to either `train` or `val`,
and train frames are the *hardest* of 3 candidate times per clip (where the fine-tuned and
COCO models disagree). Every image comes with pre-labels in `labels/*/*.txt` (YOLO format,
class 0 = cow). `manifest.csv` lists camera, clip, time, IR/day and model disagreement.

## 2. Correct the boxes in a LOCAL Label Studio

The event data must stay within the event, so use a local tool (no Roboflow/CVAT cloud).
Label Studio lives in its own venv (`.venv-labelstudio`) so it cannot change the pipeline's numpy/OpenCV.

```bash
.venv/bin/python scripts/labelstudio_bridge.py make-tasks      # tasks + pre-labels -> data/annotation/labelstudio_tasks.json
./scripts/start_labelstudio.sh                                 # http://localhost:8080
```

One-time project setup in the browser:
1. Sign up. The account is local to your machine.
2. **Create Project** (e.g. "Pen6 cows") → *Labeling Setup* → *Custom template* → *Code* → paste
   `data/annotation/labelstudio_config.xml` → **Save**.
3. Project **Settings → Cloud Storage → Add Source Storage** → type **Local files**, absolute path
   `<repo>/data/annotation/images` → **Add Storage**. Do **not** press *Sync*: the tasks come from the JSON.
4. **Settings → Annotation**: turn on **Use predictions to prelabel tasks** (model `prelabel_yolo26_ensemble`).
5. **Import** → upload `data/annotation/labelstudio_tasks.json`.

Then **Label All Tasks**. For each frame, fix the boxes and **Submit** (Ctrl/Cmd+Enter).
- Draw: press `1` (cow), then drag. Resize: drag a corner. Delete: click a box, then `Backspace`.
- Empty frame: just Submit with no boxes.
- Zoom: scroll with Ctrl/Cmd held.

Export whenever you like (partial is fine): **Export → JSON**, then

```bash
.venv/bin/python scripts/labelstudio_bridge.py to-yolo --export ~/Downloads/<export>.json --only-reviewed
```

This writes your boxes into `data/annotation/labels/` and creates `dataset_reviewed.yaml`
(reviewed frames only). Use it in place of `dataset.yaml` in step 3.

Rules:
1. **Box every cow you can see**, in any pen. Pen membership is decided later by the pen ROI,
   so the detector must learn that every cow is a cow.
2. **Include cows cut off by the frame edge** (rear views at the bunk in cam1/cam2):
   box the visible part to the edge.
3. **Include partly hidden cows** in clusters when roughly ≥ 30% of the body is visible.
   Box the visible extent, not a guess of the hidden body.
4. **Tight boxes**: body + head + legs; ignore the tail tip.
5. **Do not box** people, dogs, birds or reflections. Delete pre-labels on them.
6. Night/IR frames count the same; label what you can see.
7. Delete duplicate pre-labels (two boxes on one cow) and split boxes that cover two cows.

## 3. Train (boxes only, high resolution)

```bash
yolo detect train model=yolo26m.pt data=data/annotation/dataset_reviewed.yaml \
     imgsz=1280 epochs=80 batch=8 patience=20 close_mosaic=10 \
     project=runs/detect name=angus_handlabeled
```

- `imgsz=1280` matters. At 640 a 4000 px frame shrinks 6×, and far-field and edge-cut cows disappear.
- Optionally also add the existing pseudo-labeled set (`data/training`) to `train:` for more
  volume. Keep `val:` hand-labeled only so the score is honest.

## 4. Evaluate what matters

```bash
yolo detect val model=runs/detect/angus_handlabeled/weights/best.pt data=data/annotation/dataset_reviewed.yaml imgsz=1280
```

Report recall overall **and** on cam1/cam2 bunk frames and night frames (filter by
`manifest.csv`). Then run the pipeline with the new weights:

```bash
python main.py ... --model runs/detect/angus_handlabeled/weights/best.pt --imgsz 1280
```

## 5. Pen ROI, feed bunk and waterer zones (5 min per camera)

```bash
python scripts/draw_zones.py --camera cam1 --video "input/Day 1/Cam 01/Event20261107071945006.Avi" --t 30
python scripts/draw_zones.py --camera cam2 --video "input/Day 1/Cam 02/Event20261107181537017.Avi" --t 30
python scripts/draw_zones.py --camera cam3 --video "input/Day 1/Cam 03/Cam 03/Event20261107071935003.Avi" --t 30
```

Keys: `1` pen ROI, `2` feed bunk, `3` waterer; click to add points, `u` undo, `c` clear, `s` save, `q` quit.
Zones are saved to `configs/camera_zones.yaml` and override the estimated ROIs and the metric zones:
- **feeding** = at least 12% of the cow's box lies in the feed-bunk band (head/neck in the bunk)
- **drinking** = at least 5% of the box lies in the waterer polygon.

Check the result with `python scripts/calibration_overlay.py --camera cam1 --video ... --out outputs/calibration/cam1_overlay.jpg`.

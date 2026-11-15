"""Create a tiny synthetic demo: 2-minute night-ish/day clip + a YOLO CSV in the expected format, to try the pipeline end to end
without real data:   python scripts/make_demo_video.py demo/  &&  see README 'Try it on the demo'."""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo")
name = "Event20261107033930006.Avi"
(out / "videos" / "cam_a").mkdir(parents=True, exist_ok=True)
W, H, FPS, SECS = 320, 240, 10, 120
vw = cv2.VideoWriter(str(out / "videos" / "cam_a" / name), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
for i in range(FPS * SECS):
    t = i / FPS
    f = np.full((H, W, 3), 22, np.uint8)                       # dark frame: night
    x = int(40 + t)
    cv2.rectangle(f, (x, 80), (x + 80, 160), (70, 70, 70), -1)
    cv2.rectangle(f, (200, 60), (280, 90), (22 + (80 if int(t * 2) % 2 == 0 else 0),) * 3, -1)   # head moves at 1 Hz
    cv2.rectangle(f, (200, 90), (280, 140), (60, 60, 60), -1)
    vw.write(f)
vw.release()
with open(out / "yolo_output.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["file", "timestamp", "cow_id", "behavior", "confidence", "x1", "y1", "x2", "y2"])
    for s in range(SECS):
        x = int(40 + s)
        w.writerow([name, s, "C01", "walking", 0.9, x, 80, x + 80, 160])
        w.writerow([name, s, "C02", "standing", 0.95, 200, 60, 280, 140])      # actually ruminating: the agent should correct this
print("demo written to", out)

# Test the flow on the real sample clip (Event20261107152517006.Avi)

1. Put the clip here:  `videos/back/Event20261107152517006.Avi`   (the folder name decides the camera; `back` is a guess, rename if needed)
2. From the project root (Windows PowerShell: set `$env:PYTHONPATH="src"` instead of `export`):

```
export PYTHONPATH=src
python -m brd_agent.dbcli --store test_store import-yolo --yolo examples/real_clip/yolo_output.csv --video-root videos
python -m brd_agent.dbcli --store test_store aggregate --video-root videos --cows C01,C02,C03,C04,C05,C06,C07,C08,C09,C10
python -m brd_agent.dbcli --store test_store run --backend video --video-root videos --pen-map examples/real_clip/pen_map.yaml --vlm mock --cows C05,C07 --limit-per-cow 2 --trace-dir out/traces
python -m brd_agent.dbcli --store test_store export --out out/level1
```
Step 3 with `--vlm mock` checks the plumbing only (video reading, crops, measurements, store, traces). For real judgements set `.env`
(`USE_LOCAL_VLM=false`, `OPENAI_API_KEY`, `OPENAI_MODEL`) and use `--vlm auto`.

What is in the files
* `yolo_output.csv`: FAKE YOLO output for the clip: 7 cows, boxes hand-placed from one frame (static, so speed is 0), one row per cow per second.
  Two deliberate YOLO errors to see the agent catch: **C05** is labelled `standing` but is lying; **C07** is labelled `walking` but is at the feeder.
* `pen_map.yaml`: rough pixel polygons (feed bunk, resting) in native 4000x3000 coordinates. Redraw for real cameras.
* `cows.csv`: what C01..C10 mean. IDs are just stable labels; this table records which physical cow each one is.

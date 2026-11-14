"""Activity-Density Stratified Video Sampler and Feed Bunk/Waterer Crowding Selector.

Scans 5-minute video clips, evaluates crowding in Feeding and Drinking zones,
generates a visual contact-sheet review interface, and adaptively samples dense
frames (1 FPS) for crowded active periods and sparse frames for resting periods.
"""

import os
import glob
import time
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np

from ..video.video_reader import FrameReader
from ..calibration.pen_boundary import PenBoundaryManager
from ..calibration.camera_calibration import CameraCalibrator
from ..calibration.coordinate_transform import CoordinateTransformer
from ..detection.cow_detector import CowDetector


class ActivitySampler:
    """Identifies crowded feeding/drinking periods across video clips for targeted dataset creation."""

    def __init__(
        self,
        model_path: str = "yolo26x-seg.pt",
        output_dir: str = "outputs/activity_sampling",
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "thumbnails"), exist_ok=True)

        self.boundary_manager = PenBoundaryManager()
        self.calibrator = CameraCalibrator()
        self.transformer = CoordinateTransformer(self.calibrator)
        self.detector = CowDetector(model_path=model_path, boundary_manager=self.boundary_manager)

    def scan_clip(
        self,
        video_path: str,
        camera_id: str,
        probe_interval_sec: float = 45.0,
    ) -> Dict[str, Any]:
        """
        Fast-probes a 5-minute clip to measure crowding at the feed bunk and waterer.
        Takes ~1-2 seconds per 5-minute video.
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        reader = FrameReader(video_path, target_fps=1.0 / max(1.0, probe_interval_sec))
        
        probe_records = []
        best_thumbnail = None
        max_bunk_cows = 0
        max_waterer_cows = 0
        max_total_cows = 0

        for f_idx, ts, frame in reader.frames():
            h, w = frame.shape[:2]
            dets, _ = self.detector.detect(
                frame_bgr=frame,
                camera_id=camera_id,
                frame_number=f_idx,
                timestamp_sec=ts,
                filter_outside_pen=True,
            )

            bunk_count = 0
            waterer_count = 0

            for d in dets:
                cx = (d.bbox[0] + d.bbox[2]) / 2.0
                y2 = d.bbox[3]
                world_pos = self.transformer.pixel_to_world(camera_id, cx, y2, current_res=(w, h))
                if world_pos:
                    wx, wy = world_pos
                    # Feeding zone: Y <= 4.0m
                    if wy <= 4.0:
                        bunk_count += 1
                    # Drinking zone: X >= 9.0m, Y >= 20.0m
                    elif wx >= 9.0 and wy >= 20.0:
                        waterer_count += 1

            total_in_pen = len(dets)
            if bunk_count > max_bunk_cows or best_thumbnail is None:
                max_bunk_cows = max(max_bunk_cows, bunk_count)
                # Generate thumbnail with detections drawn
                thumb = cv2.resize(frame, (640, 480))
                # Annotate thumbnail
                for d in dets:
                    bx1 = int(d.bbox[0] * 640 / w)
                    by1 = int(d.bbox[1] * 480 / h)
                    bx2 = int(d.bbox[2] * 640 / w)
                    by2 = int(d.bbox[3] * 480 / h)
                    cv2.rectangle(thumb, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(thumb, f"{base_name} | Bunk: {bunk_count} | Total: {total_in_pen}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                best_thumbnail = thumb

            max_waterer_cows = max(max_waterer_cows, waterer_count)
            max_total_cows = max(max_total_cows, total_in_pen)

            probe_records.append({
                "ts": ts,
                "total_cows": total_in_pen,
                "bunk_cows": bunk_count,
                "waterer_cows": waterer_count,
            })

        # Save thumbnail image
        thumb_filename = f"{camera_id}_{base_name}_thumb.jpg"
        thumb_path = os.path.join(self.output_dir, "thumbnails", thumb_filename)
        if best_thumbnail is not None:
            cv2.imwrite(thumb_path, best_thumbnail)

        # Classification recommendation: Crowded if >= 2 cows at bunk or waterer
        is_crowded = (max_bunk_cows >= 2 or max_waterer_cows >= 2)

        return {
            "video_path": video_path,
            "clip_id": base_name,
            "camera_id": camera_id,
            "duration_sec": reader.duration_sec,
            "max_total_cows": max_total_cows,
            "max_bunk_cows": max_bunk_cows,
            "max_waterer_cows": max_waterer_cows,
            "is_crowded_recommendation": is_crowded,
            "thumbnail_path": thumb_path,
            "relative_thumb_url": f"thumbnails/{thumb_filename}",
            "probe_records": probe_records,
        }

    def generate_html_review_dashboard(
        self,
        clip_results: List[Dict[str, Any]],
        html_filename: str = "activity_clip_selector.html",
    ) -> str:
        """Generates an interactive HTML dashboard showing clip thumbnails and crowding recommendations."""
        html_path = os.path.join(self.output_dir, html_filename)

        cards_html = ""
        for i, clip in enumerate(clip_results):
            crowded_badge = (
                '<span class="badge bg-danger">CROWDED (Active)</span>'
                if clip["is_crowded_recommendation"]
                else '<span class="badge bg-secondary">QUIESCENT (Resting)</span>'
            )
            checked = "checked" if clip["is_crowded_recommendation"] else ""

            cards_html += f"""
            <div class="col-md-4 mb-4">
              <div class="card shadow-sm h-100">
                <img src="{clip['relative_thumb_url']}" class="card-img-top" alt="Thumbnail" style="height: 220px; object-fit: cover;">
                <div class="card-body">
                  <h6 class="card-title text-truncate" title="{clip['clip_id']}">{clip['clip_id']}</h6>
                  <p class="card-text mb-1">{crowded_badge} <span class="badge bg-info text-dark">{clip['camera_id'].upper()}</span></p>
                  <p class="card-text small text-muted mb-2">
                    Feed Bunk Cows: <strong>{clip['max_bunk_cows']}</strong> | Waterer Cows: <strong>{clip['max_waterer_cows']}</strong><br>
                    Max Visible: <strong>{clip['max_total_cows']}</strong> / 10 | Duration: {clip['duration_sec']:.0f}s
                  </p>
                  <div class="form-check form-switch">
                    <input class="form-check-input clip-toggle" type="checkbox" id="clip_{i}" data-path="{clip['video_path']}" {checked}>
                    <label class="form-check-label fw-bold" for="clip_{i}">Sample Densely (1 FPS)</label>
                  </div>
                </div>
              </div>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Feed Bunk & Waterer Activity Selector</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {{ background: #f8f9fa; padding: 20px; }}
    .card:hover {{ transform: translateY(-3px); transition: 0.2s; }}
  </style>
</head>
<body>
  <div class="container-fluid">
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div>
        <h2>Texas A&M Feedlot Pen 6: Activity-Driven Video Sampler</h2>
        <p class="text-muted mb-0">Select high-activity clips (crowded feed bunk/waterer) to sample densely (1 FPS) for training YOLO26.</p>
      </div>
      <div>
        <button class="btn btn-outline-primary me-2" onclick="selectAll(true)">Select All Crowded</button>
        <button class="btn btn-outline-secondary me-2" onclick="selectAll(false)">Clear All</button>
        <button class="btn btn-success fw-bold" onclick="exportSelection()">Save & Export Dataset Plan</button>
      </div>
    </div>
    <div class="row">
      {cards_html}
    </div>
  </div>
  <script>
    function selectAll(crowdedOnly) {{
      document.querySelectorAll('.clip-toggle').forEach(el => {{
        if (crowdedOnly) {{
          const isCrowded = el.closest('.card-body').innerText.includes('CROWDED');
          el.checked = isCrowded;
        }} else {{
          el.checked = false;
        }}
      }});
    }}
    function exportSelection() {{
      const selected = [];
      document.querySelectorAll('.clip-toggle:checked').forEach(el => {{
        selected.push(el.getAttribute('data-path'));
      }});
      alert('Selected ' + selected.length + ' active clips for dense 1 FPS extraction! Run the extractor to construct training dataset.');
    }}
  </script>
</body>
</html>
"""
        with open(html_path, "w") as f:
            f.write(html_content)

        return html_path

    def extract_activity_dataset(
        self,
        clip_evaluations: List[Dict[str, Any]],
        output_train_dir: str = "data/training",
        dense_stride_sec: float = 2.0,   # 1 frame every 2s during crowded periods
        sparse_stride_sec: float = 60.0, # 1 frame every 60s during resting periods
    ) -> Dict[str, int]:
        """
        Extracts frames adaptively: dense frames when cows are active/crowded, sparse when resting.
        """
        train_img_dir = os.path.join(output_train_dir, "images", "train")
        val_img_dir = os.path.join(output_train_dir, "images", "val")
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)

        total_extracted = 0
        train_count = 0
        val_count = 0

        for clip in clip_evaluations:
            v_path = clip["video_path"]
            cam_id = clip["camera_id"]
            is_dense = clip["is_crowded_recommendation"]

            stride = dense_stride_sec if is_dense else sparse_stride_sec
            reader = FrameReader(v_path, target_fps=1.0 / stride)

            clip_frames = []
            for f_idx, ts, frame in reader.frames():
                name = f"{cam_id}_{clip['clip_id']}_f{f_idx:06d}"
                # Optimize resolution to 1280px max dimension for fast GPU DataLoader throughput
                h, w = frame.shape[:2]
                if w > 1280 or h > 960:
                    scale = 1280.0 / max(w, h)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                clip_frames.append((frame, name))

            # 80/20 train/val split per clip
            n_c = len(clip_frames)
            n_train = max(1, int(n_c * 0.80))

            for frame, name in clip_frames[:n_train]:
                cv2.imwrite(os.path.join(train_img_dir, f"{name}.jpg"), frame)
                train_count += 1

            for frame, name in clip_frames[n_train:]:
                cv2.imwrite(os.path.join(val_img_dir, f"{name}.jpg"), frame)
                val_count += 1

            total_extracted += n_c
            print(f"[{cam_id.upper()}] {clip['clip_id']}: {'DENSE' if is_dense else 'SPARSE'} ({n_c} frames extracted)")

        return {
            "total_extracted": total_extracted,
            "train_count": train_count,
            "val_count": val_count,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Feed Bunk & Waterer Activity Sampler")
    parser.add_argument("--source", type=str, default="input/Day 1/Cam 01", help="Video file or directory")
    parser.add_argument("--camera", type=str, default="cam1", help="Camera ID (cam1, cam2, cam3)")
    parser.add_argument("--max-clips", type=int, default=10, help="Maximum clips to probe")
    parser.add_argument("--probe-stride", type=float, default=60.0, help="Seconds between probe frames")
    parser.add_argument("--dense-stride", type=float, default=2.0, help="Seconds between frames when crowded")
    parser.add_argument("--sparse-stride", type=float, default=60.0, help="Seconds between frames when resting")
    parser.add_argument("--extract", action="store_true", help="Extract dataset frames immediately after scan")

    args = parser.parse_args()
    sampler = ActivitySampler()

    if os.path.isdir(args.source):
        files = sorted(glob.glob(os.path.join(args.source, "*.Avi")) + glob.glob(os.path.join(args.source, "*.avi")) + glob.glob(os.path.join(args.source, "*", "*.Avi")))
    else:
        files = [args.source]

    target_files = files[:args.max_clips]
    print(f"================================================================================")
    print(f"SCANNING {len(target_files)} CLIPS FOR BUNK & WATERER ACTIVITY ({args.camera.upper()})")
    print(f"================================================================================")

    results = []
    for f in target_files:
        res = sampler.scan_clip(f, args.camera, probe_interval_sec=args.probe_stride)
        status = "CROWDED (Active)" if res["is_crowded_recommendation"] else "QUIESCENT (Resting)"
        print(f"  {res['clip_id']}: Bunk={res['max_bunk_cows']} | Waterer={res['max_waterer_cows']} | {status}")
        results.append(res)

    html_path = sampler.generate_html_review_dashboard(results)
    print(f"\n[DONE] Generated Visual Review Dashboard: {html_path}")

    if args.extract:
        print("\nExtracting activity-stratified training frames...")
        stats = sampler.extract_activity_dataset(
            results,
            dense_stride_sec=args.dense_stride,
            sparse_stride_sec=args.sparse_stride,
        )
        print(f"[EXTRACTED] Total: {stats['total_extracted']} (Train: {stats['train_count']}, Val: {stats['val_count']})")


if __name__ == "__main__":
    main()

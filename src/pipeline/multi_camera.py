"""Multi-camera synchronized cattle tracking, cross-camera identity association, and dashboard pipeline."""

import os
import time
from datetime import timedelta
from typing import Dict, Optional, Any, List, Tuple
import cv2
import numpy as np

from ..video.video_reader import FrameReader, open_reader
from ..video.clip_sequence import is_clip_folder, list_clips, parse_wall
from ..calibration.pen_boundary import PenBoundaryManager
from ..calibration.camera_calibration import CameraCalibrator
from ..calibration.coordinate_transform import CoordinateTransformer
from ..calibration.zone_mapping import ZoneMapper
from ..detection.cow_detector import CowDetector
from ..tracking.local_tracker import ByteTracker, LocalTrack
from ..tracking.reidentification import CowAppearanceExtractor
from ..tracking.appearance_gallery import AppearanceGalleryManager
from ..tracking.global_identity import GlobalIdentityManager
from ..tracking.tracklet_identity import TrackletIdentityManager, color_descriptor, fuse_embedding
from ..visualization.annotation import VisualAnnotator
from ..visualization.tracking_dashboard import TrackingDashboard
from ..storage.database import TrackingDatabase, TrackingRecord
from ..storage.export import TrackingExporter
from ..evaluation.evaluation_report import EvaluationReportGenerator
from .checkpoint import Checkpointer, log_error
from ..detection.clinical_visibility import ClinicalVisibilityAssessor
from ..detection.clinical_head_detector import (
    BovineHeadClinicalAnalyzer,
    ClinicalWindowAggregator,
)


class MultiCameraPipeline:
    """Synchronous 3-camera cattle tracking and global identity fusion pipeline."""

    def __init__(
        self,
        camera_sources: Dict[str, str], # 'cam1': path, 'cam2': path, 'cam3': path
        model_path: str = "yolo26x-seg.pt",
        confidence_threshold: float = 0.20,
        target_fps: Optional[float] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        output_dir: str = "outputs",
        enable_reid: bool = True,
        id_gallery_path: Optional[str] = "outputs/id_gallery.npz",
        legacy_identity: bool = False,
        imgsz: int = 640,
        wall_start: Optional[str] = None,
        wall_end: Optional[str] = None,
        record_assets: bool = True,
        checkpoint_sec: float = 900.0,
        max_consecutive_errors: int = 30,
    ):
        self.camera_sources = camera_sources
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.target_fps = target_fps
        self.max_consecutive_errors = max_consecutive_errors
        # checkpoints only for clip folders (the pipeline's case): there a stream can restart at any wall time
        self.ckpt = None
        resume = None
        if checkpoint_sec and checkpoint_sec > 0 and any(is_clip_folder(src) for src in camera_sources.values()):
            fp = dict(sources=sorted((c, os.path.abspath(s)) for c, s in camera_sources.items()),
                      model=os.path.basename(model_path), conf=confidence_threshold, fps=target_fps, imgsz=imgsz,
                      start_sec=start_sec, end_sec=end_sec, wall_start=str(wall_start), wall_end=str(wall_end),
                      reid=enable_reid, legacy=legacy_identity, assets=record_assets)
            self.ckpt = Checkpointer(output_dir, fp, checkpoint_sec)
            resume = self.ckpt.load()
        self.assets = None
        if resume is not None:
            self.assets = resume["assets"]
            if self.assets is not None:
                self.assets.reopen(resume["asset_sizes"])
        elif record_assets:  # Level 2 risk assets (gallery, motion, bunk signal): src/assets/recorder.py
            from ..assets.recorder import AssetRecorder
            self.assets = AssetRecorder(output_dir)

        # Core components
        self.boundary_manager = PenBoundaryManager()
        self.calibrator = CameraCalibrator()
        self.transformer = CoordinateTransformer(self.calibrator)
        self.zone_mapper = ZoneMapper()

        self.detector = CowDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            boundary_manager=self.boundary_manager,
            imgsz=imgsz,
        )

        self.trackers: Dict[str, ByteTracker] = {
            cam_id: ByteTracker(camera_id=cam_id) for cam_id in camera_sources
        }

        self.annotator = VisualAnnotator(draw_masks=True, draw_roi=True)
        self.dashboard = TrackingDashboard()
        self.db = TrackingDatabase()
        self.exporter = TrackingExporter(output_dir=output_dir)
        self.report_generator = EvaluationReportGenerator(output_dir=output_dir)
        self.clinical_assessor = ClinicalVisibilityAssessor()
        self.head_analyzer = BovineHeadClinicalAnalyzer(
            crops_output_dir=os.path.join(output_dir, "clinical_crops")
        )
        self.window_aggregator = ClinicalWindowAggregator(window_size_sec=10.0)

        # ReID & Global Identity
        self.enable_reid = enable_reid
        self.reid_extractor = (
            CowAppearanceExtractor(model_name="dinov2_vits14") if enable_reid else None
        )
        self.gallery = AppearanceGalleryManager()
        # Tracklet-level identity with unique per-frame assignment and a saved gallery
        # (shared across runs so COW_xx stays the same cow across clips and days).
        self.legacy_identity = legacy_identity
        if legacy_identity:
            self.global_identity_manager = GlobalIdentityManager(
                transformer=self.transformer,
                gallery=self.gallery,
            )
        else:
            self.global_identity_manager = TrackletIdentityManager(
                transformer=self.transformer, gallery_path=id_gallery_path,
            )

        # Initialize readers
        # Folders of Event clips are read as ONE continuous stream per camera on a shared
        # wall-clock grid (same origin for every camera), so tracks survive file boundaries
        # and the cameras stay time-aligned even though their clip names differ by ~10 s.
        seq_kwargs = {}
        if any(is_clip_folder(src) for src in camera_sources.values()):
            ws = parse_wall(wall_start)
            origin = ws or min(list_clips(src)[0].start
                               for src in camera_sources.values() if is_clip_folder(src))
            seq_kwargs = dict(origin=origin, wall_start=ws, wall_end=parse_wall(wall_end),
                              emit_gaps=True)
        self._seq_kwargs, self._start_sec, self._end_sec = seq_kwargs, start_sec, end_sec
        self._reopens: Dict[str, int] = {}
        self._last: Dict[str, Tuple[int, float]] = {}   # per camera: last (frame index, seconds since origin) read
        self.readers = {}
        for cam_id, src in camera_sources.items():
            kw = seq_kwargs if is_clip_folder(src) else {}
            self.readers[cam_id] = open_reader(src, target_fps=target_fps, start_sec=start_sec,
                                               end_sec=end_sec, **kw)
            if hasattr(self.readers[cam_id], "summary"):
                print(f"[MULTI-CAM] {cam_id}: continuous stream, {self.readers[cam_id].summary()}")
        self._segments = {c: getattr(r, "segment_id", 0) for c, r in self.readers.items()}
        self._resume_state = resume
        if resume is not None:
            self._restore(resume)

    # ------------------------------------------------------------------ checkpoint / resume
    def _tick(self) -> float:
        return 1.0 / self.target_fps if self.target_fps else 1.0 / 15.0

    def _reader_from(self, cam_id: str, after_sec: float, first_index: int, segment_id: int):
        """This camera's stream from the first tick after `after_sec` (seconds since origin), same tick grid."""
        origin = self._seq_kwargs["origin"]
        kw = dict(self._seq_kwargs, wall_start=origin + timedelta(seconds=after_sec + 0.5 * self._tick()),
                  first_index=first_index, segment_id=segment_id)
        try:
            return open_reader(self.camera_sources[cam_id], target_fps=self.target_fps, start_sec=self._start_sec,
                               end_sec=self._end_sec, **kw)
        except ValueError:  # no footage left after that time
            return _Finished(self.readers[cam_id])

    def _state(self, step: int, total_records: int, elapsed: float, n_errors: int) -> dict:
        return dict(step=step, total_records=total_records, elapsed=elapsed, n_errors=n_errors,
                    last=dict(self._last), reader_segments={c: getattr(r, "segment_id", 0) for c, r in self.readers.items()},
                    seen_segments=dict(self._segments), records=self.db.records,
                    identity=self.global_identity_manager, trackers=self.trackers, window=self.window_aggregator,
                    clinical=self.clinical_assessor, head=self.head_analyzer, gallery=self.gallery,
                    asset_sizes=self.assets.snapshot() if self.assets is not None else None, assets=self.assets)

    def _restore(self, st: dict):
        self.db.records = st["records"]
        self.global_identity_manager = st["identity"]
        self.trackers = st["trackers"]
        self.window_aggregator = st["window"]
        self.clinical_assessor = st["clinical"]
        self.head_analyzer = st["head"]
        self.gallery = st["gallery"]
        self._segments = st["seen_segments"]
        self._last = dict(st["last"])
        for cam_id, (idx, ts) in self._last.items():
            if cam_id in self.readers:
                self.readers[cam_id] = self._reader_from(cam_id, ts, idx + 1, st["reader_segments"].get(cam_id, 0))
        t = max(ts for _, ts in self._last.values()) if self._last else 0.0
        print(f"[CHECKPOINT] resuming at {t / 60:.1f} min of footage (step {st['step']}, {st['total_records']} records "
              f"kept, {st['elapsed'] / 60:.0f} min of earlier work saved)", flush=True)

    def run(self, save_dashboard_video: bool = True) -> Dict[str, Any]:
        """Executes multi-camera processing loop with cross-camera association."""
        print(f"[MULTI-CAM] Initializing pipeline across {len(self.camera_sources)} cameras")
        print(f"[MULTI-CAM] Detector: {self.model_path} | ReID: DINOv2-small")
        t0 = time.time()

        dashboard_writer = None
        dashboard_out_path = os.path.join(self.output_dir, "multicam_tracking_dashboard.mp4")

        # Frame generators
        frame_iters = {cam_id: reader.frames() for cam_id, reader in self.readers.items()}

        step = 0
        total_records = 0
        n_errors = 0
        consecutive = 0
        elapsed_before = 0.0
        if self._resume_state is not None:
            st = self._resume_state
            step, total_records, n_errors, elapsed_before = st["step"], st["total_records"], st["n_errors"], st["elapsed"]
            self._resume_state = None
        step0 = step
        ckpt = self.ckpt if (self.ckpt is not None and not save_dashboard_video) else None  # an mp4 cannot be resumed
        next_ckpt = (max(ts for _, ts in self._last.values()) if self._last else None)
        next_ckpt = None if next_ckpt is None else next_ckpt + (ckpt.every_sec if ckpt else 0)

        while True:
            current_frames: Dict[str, np.ndarray] = {}
            current_indices: Dict[str, int] = {}
            current_timestamps: Dict[str, float] = {}

            # Pull next frame from each camera stream
            exhausted_all = True
            for cam_id in list(frame_iters):
                try:
                    f_idx, f_ts, frame = next(frame_iters[cam_id])
                    exhausted_all = False
                except StopIteration:
                    continue
                except Exception as e:  # a clip that cannot be decoded: skip a minute of this camera, keep going
                    n_errors += 1
                    log_error(self.output_dir, f"reading {cam_id}", e)
                    k = self._reopens[cam_id] = self._reopens.get(cam_id, 0) + 1
                    idx, ts = self._last.get(cam_id, (-1, None))
                    if k > 20 or ts is None or not self._seq_kwargs:
                        print(f"[MULTI-CAM] {cam_id}: stream failed ({e!r}); camera dropped, see step_errors.log", flush=True)
                        continue
                    print(f"[MULTI-CAM] {cam_id}: read error ({e!r}); skipping 60 s of this camera", flush=True)
                    self.readers[cam_id] = self._reader_from(cam_id, ts + 60.0, idx + 1,
                                                             getattr(self.readers[cam_id], "segment_id", 0))
                    frame_iters[cam_id] = self.readers[cam_id].frames()
                    self._last[cam_id] = (idx, ts + 60.0)
                    exhausted_all = False
                    continue
                self._last[cam_id] = (f_idx, f_ts)
                if frame is None:  # no footage from this camera at this tick
                    continue
                seg = getattr(self.readers[cam_id], "segment_id", 0)
                if seg != self._segments[cam_id]:  # hole in the footage: don't bridge it with IoU
                    self.trackers[cam_id].reset()
                    self._segments[cam_id] = seg
                    if self.assets is not None:
                        self.assets.reset_camera(cam_id)
                current_frames[cam_id] = frame
                current_indices[cam_id] = f_idx
                current_timestamps[cam_id] = f_ts

            if exhausted_all:
                break
            if not current_frames:
                continue

            t_now = max(current_timestamps.values())
            try:
                master_timestamp = max(current_timestamps.values()) if current_timestamps else step * 0.1

                # 1. Detection and within-camera tracking for each camera
                camera_active_tracks: Dict[str, List[LocalTrack]] = {}
                camera_detections: Dict[str, List[Any]] = {}
                annotated_camera_frames: Dict[str, np.ndarray] = {}
                track_embeddings: Dict[Tuple[str, int], np.ndarray] = {}

                for cam_id, frame in current_frames.items():
                    h, w = frame.shape[:2]
                    f_idx = current_indices[cam_id]
                    f_ts = current_timestamps[cam_id]

                    dets, assessment = self.detector.detect(
                        frame_bgr=frame,
                        camera_id=cam_id,
                        frame_number=f_idx,
                        timestamp_sec=f_ts,
                        filter_outside_pen=True,
                    )
                    camera_detections[cam_id] = dets

                    tracks = self.trackers[cam_id].update(
                        detections=dets,
                        frame_number=f_idx,
                        timestamp_sec=f_ts,
                    )
                    camera_active_tracks[cam_id] = tracks
                    if self.assets is not None:
                        rd = self.readers[cam_id]
                        self.assets.frame(cam_id, frame, getattr(rd, "current_clip", None) or os.path.basename(rd.source),
                                          getattr(rd, "clip_offset_sec", f_ts))
                        for t in tracks:
                            self.assets.track(cam_id, t.track_id, t.bbox)

                    # Extract DINOv2 visual embeddings for active tracks
                    if self.reid_extractor is not None:
                        for t in tracks:
                            emb = self.reid_extractor.extract(frame, t.bbox)
                            if not self.legacy_identity:
                                emb = fuse_embedding(emb, color_descriptor(frame, t.bbox))
                            track_embeddings[(cam_id, t.track_id)] = emb

                # 2. Cross-camera identity association and Global Registry update
                id_kwargs = {}
                if not self.legacy_identity:  # world positions need the true frame size
                    id_kwargs["frame_sizes"] = {c: (f.shape[1], f.shape[0]) for c, f in current_frames.items()}
                self.global_identity_manager.associate_tracks(
                    camera_tracks=camera_active_tracks,
                    track_embeddings=track_embeddings,
                    current_timestamp=master_timestamp,
                    **id_kwargs,
                )

                # 3. Store structured records and prepare positions for pen map
                cow_positions_for_map: List[Dict[str, Any]] = []

                for cam_id, tracks in camera_active_tracks.items():
                    frame = current_frames[cam_id]
                    h, w = frame.shape[:2]
                    f_idx = current_indices[cam_id]
                    f_ts = current_timestamps[cam_id]
                    dets = camera_detections.get(cam_id, [])

                    for t in tracks:
                        cx = (t.bbox[0] + t.bbox[2]) / 2.0
                        bottom_y = t.bbox[3]
                        world_pos = self.transformer.pixel_to_world(
                            cam_id, cx, bottom_y, current_res=(w, h)
                        )
                        t.world_pos = world_pos

                        matching_det = None
                        for d in dets:
                            dcx = (d.bbox[0] + d.bbox[2]) / 2.0
                            dcy = (d.bbox[1] + d.bbox[3]) / 2.0
                            if abs(dcx - cx) < 30 and abs(dcy - ((t.bbox[1] + t.bbox[3]) / 2.0)) < 30:
                                matching_det = d
                                break

                        mask_ref = "POLYGON_MASK" if (matching_det and matching_det.mask) else None
                        vis_status = matching_det.visibility_status if matching_det else "VISIBLE"
                        occ_score = matching_det.occlusion_score if matching_det else 0.0

                        # Evaluate clinical head/discharge visibility metadata
                        clinical_meta = self.clinical_assessor.evaluate_cow(
                            camera_id=cam_id,
                            cow_id=t.global_cow_id or f"Trk_{t.track_id}",
                            full_bbox=t.bbox,
                            frame_shape=(h, w),
                            timestamp_sec=f_ts,
                            frame_number=f_idx,
                            world_pos=world_pos,
                        )

                        # Extract anatomical head crop and clinical signs
                        head_crop_res = self.head_analyzer.analyze_clinical_signs(
                            frame_bgr=frame,
                            cow_bbox=t.bbox,
                            camera_id=cam_id,
                            cow_id=t.global_cow_id or f"Trk_{t.track_id}",
                            timestamp_sec=f_ts,
                            frame_number=f_idx,
                            save_crop=(step == 0 or clinical_meta.nasal_region_resolvable),
                        )
                        if head_crop_res is not None:
                            self.window_aggregator.add_observation(head_crop_res)

                        reader = self.readers[cam_id]
                        record = TrackingRecord(
                            # wall_time = Event time of source_video + timestamp (offset in that clip)
                            timestamp=getattr(reader, "clip_offset_sec", f_ts),
                            camera_id=cam_id,
                            frame_number=f_idx,
                            local_track_id=t.track_id,
                            global_cow_id=t.global_cow_id,
                            bbox_x1=t.bbox[0],
                            bbox_y1=t.bbox[1],
                            bbox_x2=t.bbox[2],
                            bbox_y2=t.bbox[3],
                            center_x=cx,
                            center_y=(t.bbox[1] + t.bbox[3]) / 2.0,
                            detection_confidence=t.detection_confidence,
                            tracking_confidence=t.tracking_confidence,
                            segmentation_mask_reference=mask_ref,
                            visibility_status=vis_status,
                            occlusion_status=f"{occ_score:.2f}",
                            identity_confidence=(
                                getattr(self.global_identity_manager, "track_confidence", {}).get(
                                    (cam_id, t.track_id), 0.85)
                                if t.global_cow_id else None
                            ),
                            world_x=world_pos[0] if world_pos else None,
                            world_y=world_pos[1] if world_pos else None,
                            discharge_inspection_feasible=clinical_meta.nasal_region_resolvable,
                            best_clinical_camera=clinical_meta.best_camera_for_inspection,
                            clip_10s_reference=clinical_meta.clip_10s_reference,
                            consensus_energy=self.global_identity_manager.last_consensus_energy,
                            source_video=getattr(reader, "current_clip", None) or os.path.basename(reader.source),
                            frame_width=w,
                            frame_height=h,
                        )
                        self.db.insert(record)
                        total_records += 1

                        if world_pos:
                            cow_positions_for_map.append(
                                {
                                    "global_cow_id": t.global_cow_id or f"Trk_{t.track_id}",
                                    "world_x": world_pos[0],
                                    "world_y": world_pos[1],
                                }
                            )

                    # Annotate individual camera view
                    roi_poly = self.boundary_manager.get_pixel_polygon(cam_id, w, h)
                    ann = self.annotator.annotate(
                        frame_bgr=frame,
                        camera_id=cam_id,
                        tracks=tracks,
                        detections=dets,
                        roi_polygon=roi_poly,
                        info_text=f"Time: {f_ts:.1f}s",
                    )
                    annotated_camera_frames[cam_id] = ann

                # 4. Compose synchronized 4-panel dashboard
                dash_frame = self.dashboard.compose(
                    camera_frames=annotated_camera_frames,
                    cow_positions=cow_positions_for_map,
                    timestamp_sec=master_timestamp,
                )

                if save_dashboard_video:
                    if dashboard_writer is None:
                        dh, dw = dash_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        dashboard_writer = cv2.VideoWriter(dashboard_out_path, fourcc, 5.0, (dw, dh))
                    dashboard_writer.write(dash_frame)

                # Save snapshot of first dashboard frame
                if step == 0:
                    cv2.imwrite(
                        os.path.join(self.output_dir, "multicam_dashboard_sample.jpg"), dash_frame
                    )
                consecutive = 0
            except Exception as e:  # one bad second must not cost the chunk: log it, drop this step, continue
                n_errors += 1
                consecutive += 1
                log_error(self.output_dir, f"step {step} at {t_now / 60:.2f} min of footage", e)
                print(f"[MULTI-CAM] step {step} ({t_now / 60:.2f} min) skipped: {e!r} (details in step_errors.log)",
                      flush=True)
                if consecutive >= self.max_consecutive_errors:
                    raise RuntimeError(f"{consecutive} steps in a row failed; stopping (last: {e!r}). "
                                       f"The last checkpoint is kept: fix the cause and rerun to resume.") from e

            step += 1
            if ckpt is not None:
                if next_ckpt is None:
                    next_ckpt = t_now + ckpt.every_sec
                elif t_now >= next_ckpt:
                    took = ckpt.save(self._state(step, total_records, elapsed_before + time.time() - t0, n_errors))
                    print(f"[CHECKPOINT] saved at {t_now / 60:.0f} min of footage ({took:.1f} s)", flush=True)
                    next_ckpt = t_now + ckpt.every_sec
            if step % 300 == 0 and current_timestamps:   # a sign of life every 300 steps (5 min of footage at 1 fps)
                done_s = max(current_timestamps.values())
                rate = (step - step0) / max(0.001, time.time() - t0)
                print(f"[MULTI-CAM] progress: {done_s / 60:.0f} min of footage tracked, {rate:.2f} steps/s, "
                      f"{(time.time() - t0) / 60:.0f} min elapsed", flush=True)

        if dashboard_writer is not None:
            dashboard_writer.release()

        elapsed = time.time() - t0
        if self.assets is not None:
            print(f"[MULTI-CAM] Risk assets: {self.assets.close()} -> {self.assets.dir}")
        print(f"[MULTI-CAM] Completed {step} synchronized steps in {elapsed:.2f}s ({step / max(0.001, elapsed):.1f} steps/s)")

        # 5. Export structured tracking records (CSV & Parquet)
        export_paths = self.exporter.export_records(self.db, filename_base="tracking_records")
        print(f"[MULTI-CAM] Exported records to {export_paths['csv']}")

        # 6. Export Cow Identity Summary
        if n_errors:
            print(f"[MULTI-CAM] {n_errors} step(s)/read(s) failed and were skipped: see "
                  f"{os.path.join(self.output_dir, 'step_errors.log')}", flush=True)
        if hasattr(self.global_identity_manager, "save_gallery"):
            self.global_identity_manager.save_gallery()
            print(f"[MULTI-CAM] Saved identity gallery to {self.global_identity_manager.gallery_path}")
        cow_summary = self.global_identity_manager.get_summary()
        summary_path = self.exporter.export_cow_summary(cow_summary)
        print(f"[MULTI-CAM] Exported cow registry summary to {summary_path}")

        # 7. Generate Evaluation Report
        df = self.db.to_dataframe()
        report_path = self.report_generator.generate(
            df=df,
            model_name=self.model_path,
            runtime_sec=elapsed,
        )
        print(f"[MULTI-CAM] Generated evaluation report at {report_path}")

        # 8. Export 10-second Clinical Metadata Windows
        import json
        import pandas as pd
        clinical_records = self.window_aggregator.get_summary_records()
        clinical_json_path = os.path.join(self.output_dir, "clinical_10s_windows.json")
        clinical_csv_path = os.path.join(self.output_dir, "clinical_10s_windows.csv")
        with open(clinical_json_path, "w") as f:
            json.dump(clinical_records, f, indent=2)
        if clinical_records:
            pd.DataFrame(clinical_records).to_csv(clinical_csv_path, index=False)
        print(f"[MULTI-CAM] Exported {len(clinical_records)} 10s clinical windows to {clinical_json_path}")

        if self.ckpt is not None:   # everything is on disk now
            self.ckpt.clear()
        return {
            "total_steps": step,
            "total_records": total_records,
            "runtime_seconds": elapsed,
            "tracking_records_csv": export_paths["csv"],
            "cow_summary_json": summary_path,
            "evaluation_report_md": report_path,
            "clinical_windows_json": clinical_json_path,
            "dashboard_video": dashboard_out_path if save_dashboard_video else None,
            "skipped_errors": n_errors,
        }


class _Finished:
    """Stand-in reader for a camera whose footage ended before the resume point."""

    def __init__(self, old):
        self.source = getattr(old, "source", "")
        self.segment_id = getattr(old, "segment_id", 0)
        self.current_clip = getattr(old, "current_clip", None)
        self.clip_offset_sec = 0.0

    def frames(self):
        return iter(())

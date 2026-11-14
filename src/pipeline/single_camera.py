"""Single-camera end-to-end cattle detection, segmentation, and tracking pipeline."""

import os
import time
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

from ..video.video_reader import FrameReader, open_reader
from ..calibration.pen_boundary import PenBoundaryManager
from ..calibration.camera_calibration import CameraCalibrator
from ..calibration.coordinate_transform import CoordinateTransformer
from ..calibration.zone_mapping import ZoneMapper
from ..detection.cow_detector import CowDetector
from ..tracking.local_tracker import ByteTracker
from ..tracking.reidentification import CowAppearanceExtractor
from ..visualization.annotation import VisualAnnotator
from ..storage.database import TrackingDatabase, TrackingRecord


class SingleCameraPipeline:
    """Processes a single video feed: perception, filtering, tracking, metric projection, and annotation."""

    def __init__(
        self,
        camera_id: str,
        video_source: str,
        model_path: str = "yolo26x-seg.pt",
        confidence_threshold: float = 0.20,
        target_fps: Optional[float] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        output_dir: str = "outputs",
        enable_reid: bool = True,
    ):
        self.camera_id = camera_id
        self.video_source = video_source
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Components
        self.boundary_manager = PenBoundaryManager()
        self.calibrator = CameraCalibrator()
        self.transformer = CoordinateTransformer(self.calibrator)
        self.zone_mapper = ZoneMapper()

        self.detector = CowDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            boundary_manager=self.boundary_manager,
        )
        self.tracker = ByteTracker(camera_id=camera_id)
        self.annotator = VisualAnnotator(draw_masks=True, draw_roi=True)
        self.db = TrackingDatabase()

        self.reader = open_reader(  # a folder of Event clips -> one continuous stream
            source=video_source,
            target_fps=target_fps,
            start_sec=start_sec,
            end_sec=end_sec,
        )

        self.enable_reid = enable_reid
        self.reid_extractor = (
            CowAppearanceExtractor(model_name="dinov2_vits14") if enable_reid else None
        )

    def run(self, save_annotated_video: bool = True) -> TrackingDatabase:
        """Executes the pipeline over all frames."""
        print(f"[{self.camera_id.upper()}] Starting processing on {self.video_source}")
        t0 = time.time()

        writer = None
        video_out_path = os.path.join(self.output_dir, f"{self.camera_id}_annotated.mp4")

        processed_count = 0
        segment = getattr(self.reader, "segment_id", 0)
        for frame_idx, timestamp_sec, frame in self.reader.frames():
            if frame is None:
                continue
            if getattr(self.reader, "segment_id", 0) != segment:  # hole in the footage
                self.tracker.reset()
                segment = self.reader.segment_id
            h, w = frame.shape[:2]

            # 1. Detection & segmentation with Pen ROI filtering
            detections, assessment = self.detector.detect(
                frame_bgr=frame,
                camera_id=self.camera_id,
                frame_number=frame_idx,
                timestamp_sec=timestamp_sec,
                filter_outside_pen=True,
            )

            # 2. Local ByteTrack tracking
            active_tracks = self.tracker.update(
                detections=detections,
                frame_number=frame_idx,
                timestamp_sec=timestamp_sec,
            )

            # 3. Metric ground coordinate projection and record storage
            for t in active_tracks:
                cx = (t.bbox[0] + t.bbox[2]) / 2.0
                bottom_y = t.bbox[3]
                world_pos = self.transformer.pixel_to_world(
                    self.camera_id, cx, bottom_y, current_res=(w, h)
                )
                t.world_pos = world_pos

                # Find matching detection for mask reference and occlusion
                matching_det = None
                for d in detections:
                    # check center distance
                    dcx = (d.bbox[0] + d.bbox[2]) / 2.0
                    dcy = (d.bbox[1] + d.bbox[3]) / 2.0
                    if abs(dcx - cx) < 30 and abs(dcy - ((t.bbox[1] + t.bbox[3]) / 2.0)) < 30:
                        matching_det = d
                        break

                mask_ref = "POLYGON_MASK" if (matching_det and matching_det.mask) else None
                vis_status = matching_det.visibility_status if matching_det else "VISIBLE"
                occ_score = matching_det.occlusion_score if matching_det else 0.0

                record = TrackingRecord(
                    timestamp=getattr(self.reader, "clip_offset_sec", timestamp_sec),
                    camera_id=self.camera_id,
                    frame_number=frame_idx,
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
                    identity_confidence=0.75 if t.global_cow_id else None,
                    world_x=world_pos[0] if world_pos else None,
                    world_y=world_pos[1] if world_pos else None,
                    source_video=getattr(self.reader, "current_clip", None) or os.path.basename(self.reader.source),
                    frame_width=w,
                    frame_height=h,
                )
                self.db.insert(record)

            # 4. Annotation
            roi_poly = self.boundary_manager.get_pixel_polygon(self.camera_id, w, h)
            info = f"FPS: {self.reader.target_fps or self.reader.native_fps:.1f} | Night: {assessment['is_night']}"
            annotated_frame = self.annotator.annotate(
                frame_bgr=frame,
                camera_id=self.camera_id,
                tracks=active_tracks,
                detections=detections,
                roi_polygon=roi_poly,
                info_text=info,
            )

            # 5. Video writing
            if save_annotated_video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        video_out_path,
                        fourcc,
                        self.reader.target_fps or self.reader.native_fps,
                        (w, h),
                    )
                writer.write(annotated_frame)

            # Save sample frame snapshot
            if processed_count == 0:
                snap_path = os.path.join(self.output_dir, f"{self.camera_id}_sample_annotated.jpg")
                cv2.imwrite(snap_path, annotated_frame)

            processed_count += 1

        if writer is not None:
            writer.release()

        elapsed = time.time() - t0
        print(f"[{self.camera_id.upper()}] Processed {processed_count} frames in {elapsed:.2f}s ({processed_count / max(0.001, elapsed):.1f} FPS)")
        return self.db

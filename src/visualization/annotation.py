"""Visualization overlay for detection bounding boxes, masks, and tracking IDs."""

from typing import List, Tuple, Optional, Any, Dict
import cv2
import numpy as np


# Palette of distinctive colors for tracks/cows
COLOR_PALETTE = [
    (0, 255, 128),   # Green
    (255, 128, 0),   # Blue
    (0, 165, 255),   # Orange
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 255),   # Purple
    (255, 192, 203), # Pink
    (0, 215, 255),   # Gold
    (50, 205, 50),   # Lime
    (240, 230, 140), # Khaki
]


def get_color(identifier: int) -> Tuple[int, int, int]:
    return COLOR_PALETTE[identifier % len(COLOR_PALETTE)]


class VisualAnnotator:
    """Draws bounding boxes, instance segmentation masks, and tracking pills on frames."""

    def __init__(self, draw_masks: bool = True, draw_roi: bool = True):
        self.draw_masks = draw_masks
        self.draw_roi = draw_roi

    def annotate(
        self,
        frame_bgr: np.ndarray,
        camera_id: str,
        tracks: List[Any], # List of LocalTrack
        detections: Optional[List[Any]] = None,
        roi_polygon: Optional[np.ndarray] = None,
        info_text: Optional[str] = None,
    ) -> np.ndarray:
        """Annotates frame with tracks, masks, and metadata banner."""
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]

        # 1. Draw valid pen ROI polygon
        if self.draw_roi and roi_polygon is not None:
            cv2.polylines(canvas, [roi_polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"ROI: {camera_id.upper()} PEN BOUNDARY",
                (roi_polygon[0][0] + 10, roi_polygon[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # 2. Draw segmentation masks if detections have them
        if self.draw_masks and detections:
            overlay = canvas.copy()
            for d in detections:
                if d.mask and d.mask.polygon:
                    pts = np.array(d.mask.polygon, dtype=np.int32)
                    color = (0, 200, 255) # default mask tint
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(canvas, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

        # 3. Draw tracks and info pills
        for t in tracks:
            x1, y1, x2, y2 = [int(c) for c in t.bbox]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            color = get_color(t.track_id)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

            # Label text
            if t.global_cow_id:
                label = f"{t.global_cow_id} (Trk #{t.track_id})"
            else:
                label = f"Track #{t.track_id}"

            if t.world_pos:
                label += f" [{t.world_pos[0]:.1f}m, {t.world_pos[1]:.1f}m]"

            # Label pill background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                canvas, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 5, max(th, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            # Draw ground contact point (hooves)
            cx = int((x1 + x2) / 2)
            cv2.circle(canvas, (cx, y2), 6, (0, 0, 255), -1)

            # Draw track history trail
            if len(t.history) > 1:
                hist_pts = np.array(t.history[-25:], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [hist_pts], False, color, 2, cv2.LINE_AA)

        # 4. Top banner with camera and frame info
        banner = np.zeros((45, w, 3), dtype=np.uint8)
        text = f"{camera_id.upper()} | Active Cows: {len(tracks)}"
        if info_text:
            text += f" | {info_text}"
        cv2.putText(
            banner, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        canvas[0:45, 0:w] = cv2.addWeighted(canvas[0:45, 0:w], 0.3, banner, 0.7, 0)

        return canvas

"""Multi-camera synchronized dashboard combining 3 camera views and the 2D pen map."""

from typing import Dict, List, Optional, Any
import cv2
import numpy as np

from .pen_visualization import PenMapVisualizer


class TrackingDashboard:
    """Combines Front, Top, and Back camera views with the top-down 2D pen map."""

    def __init__(
        self,
        panel_width: int = 960,
        panel_height: int = 540,
        pen_visualizer: Optional[PenMapVisualizer] = None,
    ):
        self.pw = panel_width
        self.ph = panel_height
        self.pen_visualizer = pen_visualizer or PenMapVisualizer(
            map_width_pixels=panel_width, map_height_pixels=panel_height
        )

    def compose(
        self,
        camera_frames: Dict[str, np.ndarray], # 'cam1', 'cam2', 'cam3'
        cow_positions: List[Dict[str, Any]],
        timestamp_sec: float,
    ) -> np.ndarray:
        """
        Creates a 2x2 synchronized dashboard:
        [Front Cam 1] [Feed Bunk Cam 2]
        [Back Cam 3 ] [Top-Down 2D Pen Map]
        """
        # Canvas: 2 * pw wide by 2 * ph high
        canvas = np.zeros((self.ph * 2, self.pw * 2, 3), dtype=np.uint8)

        # 1. Front Cam 1 (Top-Left)
        if "cam1" in camera_frames and camera_frames["cam1"] is not None:
            f1 = cv2.resize(camera_frames["cam1"], (self.pw, self.ph))
        else:
            f1 = self._empty_panel("Cam 1 (Front View) - Offline")
        canvas[0 : self.ph, 0 : self.pw] = f1

        # 2. Top / Feed Bunk Cam 2 (Top-Right)
        if "cam2" in camera_frames and camera_frames["cam2"] is not None:
            f2 = cv2.resize(camera_frames["cam2"], (self.pw, self.ph))
        else:
            f2 = self._empty_panel("Cam 2 (Feed Bunk View) - Offline")
        canvas[0 : self.ph, self.pw : self.pw * 2] = f2

        # 3. Back Cam 3 (Bottom-Left)
        if "cam3" in camera_frames and camera_frames["cam3"] is not None:
            f3 = cv2.resize(camera_frames["cam3"], (self.pw, self.ph))
        else:
            f3 = self._empty_panel("Cam 3 (Back View) - Offline")
        canvas[self.ph : self.ph * 2, 0 : self.pw] = f3

        # 4. Top-Down 2D Pen Map (Bottom-Right)
        pen_map = self.pen_visualizer.render(cow_positions, timestamp_sec)
        pen_map_resized = cv2.resize(pen_map, (self.pw, self.ph))
        canvas[self.ph : self.ph * 2, self.pw : self.pw * 2] = pen_map_resized

        # Draw grid dividing lines
        cv2.line(canvas, (self.pw, 0), (self.pw, self.ph * 2), (100, 100, 100), 2)
        cv2.line(canvas, (0, self.ph), (self.pw * 2, self.ph), (100, 100, 100), 2)

        return canvas

    def _empty_panel(self, text: str) -> np.ndarray:
        panel = np.ones((self.ph, self.pw, 3), dtype=np.uint8) * 30
        cv2.putText(
            panel,
            text,
            (50, self.ph // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (150, 150, 150),
            2,
        )
        return panel

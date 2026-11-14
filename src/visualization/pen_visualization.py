"""2D bird's-eye view visualization of the physical pen layout and cow positions."""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


class PenMapVisualizer:
    """Renders a top-down metric map of Pen 6 with zones, cameras, and live cow positions."""

    def __init__(
        self,
        map_width_pixels: int = 400,
        map_height_pixels: int = 900,
        pen_width_m: float = 12.0,
        pen_length_m: float = 30.0,
    ):
        self.w_px = map_width_pixels
        self.h_px = map_height_pixels
        self.pen_w = pen_width_m
        self.pen_l = pen_length_m

        # Margins in pixels
        self.margin_x = 40
        self.margin_y = 60
        self.plot_w = self.w_px - 2 * self.margin_x
        self.plot_h = self.h_px - 2 * self.margin_y

        self.scale_x = self.plot_w / self.pen_w
        self.scale_y = self.plot_h / self.pen_l

    def world_to_map(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Converts metric (X, Y) where X in [0, 12], Y in [0, 30] to map canvas pixels."""
        px = int(self.margin_x + x_m * self.scale_x)
        # In map view: Y=0 (Front/Feed bunk) is at the bottom, Y=30 (Back) is at the top
        py = int(self.h_px - self.margin_y - y_m * self.scale_y)
        return px, py

    def render(
        self,
        cow_positions: Optional[List[Dict[str, Any]]] = None,
        timestamp_sec: float = 0.0,
    ) -> np.ndarray:
        """Draws complete 2D map with zones, landmarks, and cow markers."""
        # Dark clean aesthetic background
        canvas = np.ones((self.h_px, self.w_px, 3), dtype=np.uint8) * 35

        # 1. Pen boundary rectangle
        pt_bl = self.world_to_map(0.0, 0.0)      # Front-left
        pt_br = self.world_to_map(self.pen_w, 0.0) # Front-right
        pt_tr = self.world_to_map(self.pen_w, self.pen_l) # Back-right
        pt_tl = self.world_to_map(0.0, self.pen_l) # Back-left

        pen_poly = np.array([pt_bl, pt_br, pt_tr, pt_tl], dtype=np.int32)
        # Pen floor (light earth tint)
        cv2.fillPoly(canvas, [pen_poly], (50, 48, 45))
        # Pen boundary fence
        cv2.polylines(canvas, [pen_poly], True, (200, 200, 200), 2)

        # 2. Resting Zone
        rz_bl = self.world_to_map(1.5, 10.0)
        rz_tr = self.world_to_map(10.5, 22.0)
        cv2.rectangle(canvas, rz_bl, rz_tr, (40, 60, 45), -1)
        cv2.rectangle(canvas, rz_bl, rz_tr, (80, 140, 90), 1)
        cv2.putText(
            canvas,
            "RESTING ZONE",
            (rz_bl[0] + 30, (rz_bl[1] + rz_tr[1]) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 180, 130),
            1,
        )

        # 3. Waterer (Back right)
        w_bl = self.world_to_map(10.2, 22.0)
        w_tr = self.world_to_map(12.0, 25.0)
        cv2.rectangle(canvas, w_bl, w_tr, (200, 120, 50), -1) # Blue in BGR
        cv2.putText(
            canvas,
            "WATER",
            (w_bl[0] - 50, (w_bl[1] + w_tr[1]) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 200, 100),
            1,
        )

        # 4. Feed Bunk & Feeding Apron (Front)
        apron_bl = self.world_to_map(0.0, 0.0)
        apron_tr = self.world_to_map(self.pen_w, 3.5)
        cv2.rectangle(canvas, apron_bl, apron_tr, (55, 65, 55), -1)
        cv2.putText(
            canvas,
            "FEEDING APRON",
            (apron_bl[0] + 60, (apron_bl[1] + apron_tr[1]) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 200, 160),
            1,
        )

        bunk_bl = self.world_to_map(0.0, -1.0)
        bunk_tr = self.world_to_map(self.pen_w, 0.0)
        cv2.rectangle(canvas, bunk_bl, bunk_tr, (45, 90, 140), -1) # Brown/wood BGR
        cv2.putText(
            canvas,
            "FEED BUNK",
            (bunk_bl[0] + 70, bunk_bl[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # 5. Chute on left side
        cv2.putText(
            canvas, "CHUTE", (self.margin_x - 35, self.h_px // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1
        )

        # 6. Cameras
        # Front Cam
        fc_px = self.world_to_map(3.0, 0.0)
        cv2.circle(canvas, fc_px, 6, (0, 255, 255), -1)
        cv2.putText(canvas, "Cam 1", (fc_px[0] - 15, fc_px[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # Top Cam
        tc_px = self.world_to_map(6.0, 1.0)
        cv2.circle(canvas, tc_px, 6, (0, 165, 255), -1)
        cv2.putText(canvas, "Cam 2", (tc_px[0] - 15, tc_px[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

        # Back Cam
        bc_px = self.world_to_map(0.5, 29.5)
        cv2.circle(canvas, bc_px, 6, (255, 0, 255), -1)
        cv2.putText(canvas, "Cam 3", (bc_px[0] + 10, bc_px[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

        # 7. Draw tracked cows
        if cow_positions:
            for cow in cow_positions:
                xm = cow.get("world_x")
                ym = cow.get("world_y")
                gid = cow.get("global_cow_id", "COW")
                if xm is not None and ym is not None:
                    cpx, cpy = self.world_to_map(xm, ym)
                    # Cow circle marker
                    cv2.circle(canvas, (cpx, cpy), 8, (0, 255, 0), -1)
                    cv2.circle(canvas, (cpx, cpy), 10, (255, 255, 255), 1)
                    cv2.putText(
                        canvas,
                        gid,
                        (cpx + 12, cpy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                    )

        # 8. Title and timestamp header
        cv2.putText(
            canvas,
            "PEN 6 TOP-DOWN METRIC MAP",
            (self.margin_x - 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"Time: {timestamp_sec:.1f}s | Grid: 12m x 30m",
            (self.margin_x - 10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )

        return canvas

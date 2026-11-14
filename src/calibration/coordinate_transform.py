"""Coordinate transformation between camera image pixels and metric physical pen coordinates."""

from typing import Optional, Tuple, List
import cv2
import numpy as np
from .camera_calibration import CameraCalibrator


class CoordinateTransformer:
    """Transforms 2D pixel coordinates to metric pen coordinates (world_x, world_y) in meters."""

    def __init__(self, calibrator: CameraCalibrator):
        self.calibrator = calibrator
        # Valid pen physical bounds in meters
        self.min_x = -1.0
        self.max_x = 13.0
        self.min_y = -2.0
        self.max_y = 32.0

    def pixel_to_world(
        self,
        camera_id: str,
        pixel_x: float,
        pixel_y: float,
        current_res: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Transforms pixel coordinate (e.g. bottom-center of bounding box) to physical (X, Y) in meters.
        If current_res differs from the calibration reference resolution, rescales coordinates first.
        """
        H = self.calibrator.get_homography(camera_id)
        if H is None:
            return None

        # Rescale if needed
        cam_info = self.calibrator.camera_configs.get(camera_id, {})
        ref_res = cam_info.get("reference_resolution", [3024, 1640])
        ref_w, ref_h = ref_res[0], ref_res[1]

        if current_res is not None and (current_res[0] != ref_w or current_res[1] != ref_h):
            scale_x = ref_w / float(current_res[0])
            scale_y = ref_h / float(current_res[1])
            u = pixel_x * scale_x
            v = pixel_y * scale_y
        else:
            u, v = pixel_x, pixel_y

        pt = self.calibrator.undistort(camera_id, np.array([[u, v]])).reshape(1, 1, 2)
        transformed = cv2.perspectiveTransform(pt, H)
        world_x = float(transformed[0, 0, 0])
        world_y = float(transformed[0, 0, 1])

        # Sanity check: do not return wildly divergent points outside physical lot
        if (
            self.min_x <= world_x <= self.max_x
            and self.min_y <= world_y <= self.max_y
        ):
            return round(world_x, 2), round(world_y, 2)

        return None

    def world_to_pixel(
        self,
        camera_id: str,
        world_x: float,
        world_y: float,
        current_res: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[float, float]]:
        """Transforms physical coordinate (X, Y) in meters to camera image pixel space."""
        H_inv = self.calibrator.inv_homographies.get(camera_id)
        if H_inv is None:
            return None

        pt = np.array([[[world_x, world_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H_inv)
        u, v = self.calibrator.distort(camera_id, transformed.reshape(1, 2))[0].tolist()

        cam_info = self.calibrator.camera_configs.get(camera_id, {})
        ref_res = cam_info.get("reference_resolution", [3024, 1640])
        ref_w, ref_h = ref_res[0], ref_res[1]

        if current_res is not None and (current_res[0] != ref_w or current_res[1] != ref_h):
            scale_x = float(current_res[0]) / ref_w
            scale_y = float(current_res[1]) / ref_h
            u *= scale_x
            v *= scale_y

        return round(u, 1), round(v, 1)

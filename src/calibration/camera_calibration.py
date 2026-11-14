"""Camera calibration and homography estimation for the shared pen coordinate system."""

import os
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import yaml


class CameraCalibrator:
    """Estimates and stores planar ground homographies between camera views and physical pen floor."""

    def __init__(self, config_path: str = "configs/cameras.yaml"):
        self.config_path = config_path
        self.homographies: Dict[str, np.ndarray] = {}  # camera_id -> 3x3 H (pixel -> world)
        self.inv_homographies: Dict[str, np.ndarray] = {}  # camera_id -> 3x3 H_inv (world -> pixel)
        self.reprojection_errors: Dict[str, float] = {}
        self.camera_configs: Dict[str, Any] = {}
        self.load_and_calibrate()

    def load_and_calibrate(self):
        """Loads correspondences from cameras.yaml and calculates homographies."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Cameras config not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        self.camera_configs = data.get("cameras", {})
        for cam_id, info in self.camera_configs.items():
            pts = info.get("calibration_points", [])
            if len(pts) >= 4:
                ref_res = info.get("reference_resolution", [1920, 1080])
                ref_w, ref_h = ref_res[0], ref_res[1]

                src_list = []
                dst_pts = []
                for p in pts:
                    if "image_undistorted_px" in p:
                        # Off-frame corners (constructed from fence-line intersections) are
                        # stored directly in undistorted reference pixels.
                        src_list.append(np.array(p["image_undistorted_px"], dtype=np.float32))
                    else:
                        norm_u, norm_v = p["image"]
                        src_list.append(self.undistort(cam_id, np.array([[norm_u * ref_w, norm_v * ref_h]]))[0])
                    dst_pts.append(list(p["world"]))

                src_arr = np.array(src_list, dtype=np.float32)
                dst_arr = np.array(dst_pts, dtype=np.float32)

                H, mask = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
                if H is not None:
                    self.homographies[cam_id] = H
                    self.inv_homographies[cam_id] = np.linalg.inv(H)
                    # Compute mean reprojection error
                    projected = cv2.perspectiveTransform(src_arr.reshape(-1, 1, 2), H).reshape(-1, 2)
                    err = float(np.mean(np.linalg.norm(projected - dst_arr, axis=1)))
                    self.reprojection_errors[cam_id] = err

    # ------------------------------------------------------------------ lens model
    # One-parameter division model (Fitzgibbon 2001) in reference-resolution pixels:
    #   p_u = c + (p_d - c) / (1 + k1 * r^2),  r = |p_d - c| / half_diagonal
    # Fitted per camera so that rails that are straight in the world image straight
    # (cam1 is a fisheye: k1 ~ -0.28). k1 = 0 -> pure pinhole.
    def _lens(self, camera_id: str):
        info = self.camera_configs.get(camera_id, {})
        w, h = info.get("reference_resolution", [1920, 1080])
        c = np.array([w / 2.0, h / 2.0], dtype=np.float64)
        return float(info.get("distortion_k1", 0.0)), c, float(np.hypot(w / 2.0, h / 2.0))

    def undistort(self, camera_id: str, pts: np.ndarray) -> np.ndarray:
        k, c, R = self._lens(camera_id)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if k == 0.0:
            return pts.astype(np.float32)
        d = (pts - c) / R
        r2 = (d ** 2).sum(1, keepdims=True)
        return (c + d / (1.0 + k * r2) * R).astype(np.float32)

    def distort(self, camera_id: str, pts: np.ndarray) -> np.ndarray:
        k, c, R = self._lens(camera_id)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if k == 0.0:
            return pts.astype(np.float32)
        u = (pts - c) / R
        d = u.copy()
        for _ in range(50):  # fixed point of d = u * (1 + k |d|^2)
            d = u * (1.0 + k * (d ** 2).sum(1, keepdims=True))
        return (c + d * R).astype(np.float32)

    def has_calibration(self, camera_id: str) -> bool:
        """Returns True if a valid ground-plane homography is available."""
        return camera_id in self.homographies

    def get_homography(self, camera_id: str) -> Optional[np.ndarray]:
        return self.homographies.get(camera_id)

    def get_reprojection_error(self, camera_id: str) -> Optional[float]:
        return self.reprojection_errors.get(camera_id)

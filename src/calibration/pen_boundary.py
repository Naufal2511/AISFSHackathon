"""Pen boundary definition, ROI validation, and exclusion masking."""

import os
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import yaml


class PenBoundaryManager:
    """Manages polygonal ROIs for cameras to restrict detection/tracking to the pen."""

    def __init__(self, config_path: str = "configs/pen_layout.yaml"):
        self.config_path = config_path
        self.camera_rois: Dict[str, np.ndarray] = {}  # camera_id -> normalized Nx2 polygon
        self.trough_rois: Dict[str, np.ndarray] = {}
        self.load_config()

    def load_config(self):
        """Loads normalized polygonal coordinates from pen_layout.yaml."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Pen layout config not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        rois_data = data.get("camera_rois", {})
        for cam_id, details in rois_data.items():
            pts = details.get("normalized_polygon", [])
            if pts:
                self.camera_rois[cam_id] = np.array(pts, dtype=np.float32)

            trough = details.get("trough_polygon", [])
            if trough:
                self.trough_rois[cam_id] = np.array(trough, dtype=np.float32)

        # Hand-drawn ROIs (scripts/draw_zones.py) override the YAML estimates.
        from .camera_zones import load_zones
        drawn = load_zones(os.path.join(os.path.dirname(self.config_path) or ".", "camera_zones.yaml"))
        for cam_id, layers in drawn.items():
            if "roi" in layers:
                self.camera_rois[cam_id] = np.array(layers["roi"], dtype=np.float32)

    def get_pixel_polygon(self, camera_id: str, width: int, height: int) -> Optional[np.ndarray]:
        """Converts normalized ROI polygon to pixel coordinates for given image resolution."""
        if camera_id not in self.camera_rois:
            return None
        norm_pts = self.camera_rois[camera_id]
        scale = np.array([width, height], dtype=np.float32)
        pixel_pts = (norm_pts * scale).astype(np.int32)
        return pixel_pts

    def is_point_inside(
        self,
        camera_id: str,
        point: Tuple[float, float],
        width: int,
        height: int,
        tolerance_px: float = 5.0,
    ) -> bool:
        """
        Tests whether a point (x, y) lies inside or within tolerance of the valid pen ROI ground.
        Using cv2.pointPolygonTest with measureDist=True:
        positive: strictly inside
        zero: exactly on the boundary edge
        negative: outside, but accepted if within -tolerance_px (e.g. hooves on fence boundary)
        """
        poly = self.get_pixel_polygon(camera_id, width, height)
        if poly is None:
            return True  # If no ROI defined, accept all points

        signed_dist = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), measureDist=True)
        return signed_dist >= -tolerance_px

    def is_bbox_inside(
        self,
        camera_id: str,
        bbox: Tuple[float, float, float, float],
        width: int,
        height: int,
        use_ground_point: bool = True,
        tolerance_px: float = 5.0,
    ) -> bool:
        """
        Tests whether a detection bbox is valid within the pen ROI ground floor.
        If use_ground_point=True, tests the bottom-center point (where cow hooves touch dirt).
        Otherwise tests the bbox center.
        """
        x1, y1, x2, y2 = bbox
        if use_ground_point:
            test_pt = ((x1 + x2) / 2.0, y2)
        else:
            test_pt = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        return self.is_point_inside(camera_id, test_pt, width, height, tolerance_px=tolerance_px)

    def create_mask(self, camera_id: str, width: int, height: int) -> np.ndarray:
        """Creates a binary mask (uint8) where valid pen area is 255, outside is 0."""
        mask = np.zeros((height, width), dtype=np.uint8)
        poly = self.get_pixel_polygon(camera_id, width, height)
        if poly is not None:
            cv2.fillPoly(mask, [poly], 255)
        else:
            mask[:] = 255
        return mask

    def save_roi(self, camera_id: str, pixel_polygon: np.ndarray, width: int, height: int):
        """Allows manual/interactive calibration to save updated ROI coordinates."""
        norm_pts = pixel_polygon.astype(np.float32) / np.array([width, height], dtype=np.float32)
        self.camera_rois[camera_id] = norm_pts

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        if "camera_rois" not in data:
            data["camera_rois"] = {}
        if camera_id not in data["camera_rois"]:
            data["camera_rois"][camera_id] = {}

        data["camera_rois"][camera_id]["normalized_polygon"] = norm_pts.tolist()

        with open(self.config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=None)


def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Pen Boundary & ROI Manager")
    parser.add_argument("--visualize", action="store_true", help="Generate overlay verification images for all camera ROIs")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive click-to-annotate polygon GUI")
    parser.add_argument("--camera", type=str, default="cam1", help="Camera ID (cam1, cam2, cam3)")
    parser.add_argument("--image", type=str, default=None, help="Path to image for interactive annotation")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for visualizations")

    args = parser.parse_args()
    mgr = PenBoundaryManager()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.visualize:
        print("[ROI] Generating visual verification overlays...")
        for cam_id in ["cam1", "cam2", "cam3"]:
            pattern = f"input/{cam_id}/*.*"
            files = sorted(glob.glob(pattern))
            if not files:
                continue
            img_path = files[0]
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            poly = mgr.get_pixel_polygon(cam_id, w, h)
            if poly is not None:
                overlay = img.copy()
                cv2.polylines(overlay, [poly], isClosed=True, color=(0, 255, 0), thickness=4)
                cv2.fillPoly(overlay, [poly], color=(0, 255, 0))
                blended = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
                cv2.putText(blended, f"{cam_id.upper()} Pen 6 Valid ROI", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                out_path = os.path.join(args.output_dir, f"verified_roi_{cam_id}.jpg")
                cv2.imwrite(out_path, blended)
                print(f"[ROI] Saved {out_path}")

    elif args.interactive:
        img_path = args.image
        if not img_path:
            files = sorted(glob.glob(f"input/{args.camera}/*.*"))
            if files:
                img_path = files[0]
            else:
                print(f"Error: No images found for {args.camera}")
                return

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading {img_path}")
            return

        h, w = img.shape[:2]
        clicked_pts = []
        clone = img.copy()

        def mouse_cb(event, x, y, flags, param):
            nonlocal clone
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_pts.append((x, y))
                cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
                if len(clicked_pts) > 1:
                    cv2.line(clone, clicked_pts[-2], clicked_pts[-1], (0, 255, 0), 2)
                cv2.imshow("Pen Boundary Annotator (Click corners, 's' to save, 'r' to reset, 'q' to quit)", clone)

        win_name = "Pen Boundary Annotator (Click corners, 's' to save, 'r' to reset, 'q' to quit)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)
        cv2.setMouseCallback(win_name, mouse_cb)
        cv2.imshow(win_name, clone)

        print(f"Interactive Mode for {args.camera}: Click vertices along fence boundaries in clockwise order.")
        print("Keys: 's' = Save to configs/pen_layout.yaml | 'r' = Reset points | 'q' = Quit")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if len(clicked_pts) >= 3:
                    pts_np = np.array(clicked_pts, dtype=np.int32)
                    mgr.save_roi(args.camera, pts_np, w, h)
                    print(f"[SUCCESS] Saved {len(clicked_pts)} points for {args.camera} into configs/pen_layout.yaml")
                    break
                else:
                    print("Need at least 3 points to define a polygon.")
            elif key == ord('r'):
                clicked_pts.clear()
                clone = img.copy()
                cv2.imshow(win_name, clone)
                print("Reset points.")
            elif key == ord('q'):
                print("Exited without saving.")
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

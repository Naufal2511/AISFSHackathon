"""Anatomical Bovine Head Cropper and BRD Clinical Sign Assessment Module.

Extracts localized head, nasal, and ocular crops for cattle meeting optical
resolution criteria and generates Wisconsin BRD clinical scoring indicators
(nasal discharge, ocular discharge) along with 10-second temporal window metadata.
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


@dataclass
class HeadCropResult:
    camera_id: str
    cow_id: str
    timestamp_sec: float
    frame_number: int
    head_resolution: Tuple[int, int]               # (w, h) in pixels
    nasal_box_local: Tuple[int, int, int, int]     # (x1, y1, x2, y2) relative to head crop
    left_eye_box_local: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    right_eye_box_local: Tuple[int, int, int, int] # (x1, y1, x2, y2)
    nasal_moisture_index: float                    # 0.0 (dry/matte) to 1.0 (heavy mucus reflection)
    ocular_discharge_index: float                  # 0.0 (clear) to 1.0 (crusting/discharge)
    wisconsin_nasal_score: int                     # 0: normal, 1: small serous, 2: moderate, 3: copious mucopurulent
    wisconsin_eye_score: int                       # 0: normal, 1: mild discharge, 2: moderate, 3: severe purulent
    clinical_feasibility: str                      # 'FEASIBLE', 'MARGINAL', 'INSUFFICIENT'
    clip_10s_reference: str
    saved_crop_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "cow_id": self.cow_id,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "frame_number": self.frame_number,
            "head_resolution": list(self.head_resolution),
            "nasal_moisture_index": round(self.nasal_moisture_index, 3),
            "ocular_discharge_index": round(self.ocular_discharge_index, 3),
            "wisconsin_nasal_score": self.wisconsin_nasal_score,
            "wisconsin_eye_score": self.wisconsin_eye_score,
            "clinical_feasibility": self.clinical_feasibility,
            "clip_10s_reference": self.clip_10s_reference,
            "saved_crop_path": self.saved_crop_path,
        }


class BovineHeadClinicalAnalyzer:
    """
    Performs anatomical landmark isolation and optical discharge inspection
    for bovine respiratory disease (BRD) early triage.
    """

    def __init__(
        self,
        min_resolution_for_discharge: int = 120,
        crops_output_dir: str = "outputs/clinical_crops",
    ):
        self.min_res = min_resolution_for_discharge
        self.output_dir = crops_output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_head_crop(
        self,
        frame_bgr: np.ndarray,
        cow_bbox: Tuple[float, float, float, float],
        camera_id: str,
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Estimates the head region coordinates based on camera perspective and extracts the image crop.
        """
        h_frame, w_frame = frame_bgr.shape[:2]
        x1, y1, x2, y2 = cow_bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # Perspective geometry estimation:
        # In Cam 2 (overhead feed bunk): cattle lean forward down into the trough,
        # head is located in the bottom 45% of the bounding box.
        # In Cam 1 / Cam 3 (overview cameras): cattle stand upright, head is in top 35% of bounding box.
        if camera_id == "cam2":
            hx1 = int(max(0, x1 + bw * 0.20))
            hy1 = int(max(0, y1 + bh * 0.45))
            hx2 = int(min(w_frame, x1 + bw * 0.80))
            hy2 = int(min(h_frame, y2))
        else:
            hx1 = int(max(0, x1 + bw * 0.25))
            hy1 = int(max(0, y1))
            hx2 = int(min(w_frame, x1 + bw * 0.75))
            hy2 = int(min(h_frame, y1 + bh * 0.40))

        head_w = max(1, hx2 - hx1)
        head_h = max(1, hy2 - hy1)

        if head_w < 20 or head_h < 20:
            return None, (0, 0, 0, 0)

        crop = frame_bgr[hy1:hy2, hx1:hx2].copy()
        return crop, (hx1, hy1, hx2, hy2)

    def analyze_clinical_signs(
        self,
        frame_bgr: np.ndarray,
        cow_bbox: Tuple[float, float, float, float],
        camera_id: str,
        cow_id: str,
        timestamp_sec: float,
        frame_number: int,
        save_crop: bool = False,
    ) -> Optional[HeadCropResult]:
        """
        Analyzes bovine head crop for clinical discharge and Wisconsin BRD clinical scoring criteria.
        """
        crop, (hx1, hy1, hx2, hy2) = self.extract_head_crop(frame_bgr, cow_bbox, camera_id)
        if crop is None:
            return None

        ch, cw = crop.shape[:2]
        min_dim = min(ch, cw)

        # Classify optical feasibility
        if min_dim >= 160:
            feasibility = "FEASIBLE"
        elif min_dim >= self.min_res:
            feasibility = "MARGINAL"
        else:
            feasibility = "INSUFFICIENT"

        # Localize anatomical sub-regions (fractions of head crop)
        # Nasal/muzzle: bottom 35%
        nx1, ny1 = int(cw * 0.20), int(ch * 0.65)
        nx2, ny2 = int(cw * 0.80), ch
        nasal_box = (nx1, ny1, nx2, ny2)

        # Ocular: upper 25-55% lateral corners
        lx1, ly1 = int(cw * 0.05), int(ch * 0.25)
        lx2, ly2 = int(cw * 0.45), int(ch * 0.55)
        left_eye_box = (lx1, ly1, lx2, ly2)

        rx1, ry1 = int(cw * 0.55), int(ch * 0.25)
        rx2, ry2 = int(cw * 0.95), int(ch * 0.55)
        right_eye_box = (rx1, ry1, rx2, ry2)

        # Compute optical discharge indices on the crop
        # Angus cattle hair is black. Discharge/mucus shows high specular reflection and lightness shift.
        nasal_crop = crop[ny1:ny2, nx1:nx2]
        left_eye_crop = crop[ly1:ly2, lx1:lx2]
        right_eye_crop = crop[ry1:ry2, rx1:rx2]

        nasal_idx = self._compute_discharge_index(nasal_crop)
        eye_idx_l = self._compute_discharge_index(left_eye_crop)
        eye_idx_r = self._compute_discharge_index(right_eye_crop)
        eye_idx = max(eye_idx_l, eye_idx_r)

        # Map to Wisconsin BRD Clinical Score (Guinasso et al. / McGuirk Calf & Cattle Criteria)
        # Nasal score: 0 (Normal / none), 1 (Small unilateral), 2 (Bilateral serous), 3 (Copious mucopurulent)
        if nasal_idx < 0.25:
            wisconsin_nasal = 0
        elif nasal_idx < 0.50:
            wisconsin_nasal = 1
        elif nasal_idx < 0.75:
            wisconsin_nasal = 2
        else:
            wisconsin_nasal = 3

        # Eye score: 0 (Normal), 1 (Mild serous), 2 (Bilateral moderate), 3 (Severe purulent)
        if eye_idx < 0.25:
            wisconsin_eye = 0
        elif eye_idx < 0.50:
            wisconsin_eye = 1
        elif eye_idx < 0.75:
            wisconsin_eye = 2
        else:
            wisconsin_eye = 3

        # 10s video window reference
        window_idx = int(timestamp_sec // 10)
        clip_ref = f"{camera_id}_{cow_id}_clip_{window_idx * 10:05d}_{(window_idx + 1) * 10:05d}.mp4"

        saved_path = None
        if save_crop and feasibility != "INSUFFICIENT":
            # Draw landmark boxes on crop for inspection verification
            annotated_crop = crop.copy()
            cv2.rectangle(annotated_crop, (nx1, ny1), (nx2, ny2), (0, 165, 255), 2) # Orange for nasal
            cv2.rectangle(annotated_crop, (lx1, ly1), (lx2, ly2), (255, 255, 0), 2) # Cyan for eyes
            cv2.rectangle(annotated_crop, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            cv2.putText(
                annotated_crop,
                f"{cow_id} N:{wisconsin_nasal} E:{wisconsin_eye}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            filename = f"head_{cow_id}_{camera_id}_f{frame_number:05d}.jpg"
            saved_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(saved_path, annotated_crop)

        return HeadCropResult(
            camera_id=camera_id,
            cow_id=cow_id,
            timestamp_sec=timestamp_sec,
            frame_number=frame_number,
            head_resolution=(cw, ch),
            nasal_box_local=nasal_box,
            left_eye_box_local=left_eye_box,
            right_eye_box_local=right_eye_box,
            nasal_moisture_index=nasal_idx,
            ocular_discharge_index=eye_idx,
            wisconsin_nasal_score=wisconsin_nasal,
            wisconsin_eye_score=wisconsin_eye,
            clinical_feasibility=feasibility,
            clip_10s_reference=clip_ref,
            saved_crop_path=saved_path,
        )

    def _compute_discharge_index(self, patch: Optional[np.ndarray]) -> float:
        """
        Estimates reflectance and specular moisture index on bovine hair/mucosa.
        Angus hair is low-reflectance matte black; mucus/discharge exhibits bright specular reflectance.
        """
        if patch is None or patch.size == 0:
            return 0.0

        # Convert to LAB lightness
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Lightness threshold: pixels > 180 indicate specular moisture or mucus discharge
        high_lightness_ratio = float(np.mean(l_channel > 180))

        # Standard deviation of lightness: mucus produces high local contrast against dark hair
        l_std = float(np.std(l_channel)) / 128.0

        # Fused index
        score = 0.65 * high_lightness_ratio * 4.0 + 0.35 * min(1.0, l_std)
        return float(np.clip(score, 0.0, 1.0))


class ClinicalWindowAggregator:
    """
    Aggregates per-frame clinical observations into 10-second temporal windows
    with summary statistics for downstream VLM reasoning and veterinary triage.
    """

    def __init__(self, window_size_sec: float = 10.0):
        self.window_size = window_size_sec
        # (window_idx, cow_id) -> list of HeadCropResult
        self.windows: Dict[Tuple[int, str], List[HeadCropResult]] = {}

    def add_observation(self, result: HeadCropResult):
        if result is None:
            return
        w_idx = int(result.timestamp_sec // self.window_size)
        key = (w_idx, result.cow_id)
        if key not in self.windows:
            self.windows[key] = []
        self.windows[key].append(result)

    def get_summary_records(self) -> List[Dict[str, Any]]:
        """Produces aggregated 10-second clinical metadata records."""
        records = []
        for (w_idx, cow_id), obs in sorted(self.windows.items()):
            t_start = w_idx * self.window_size
            t_end = t_start + self.window_size
            feasible_obs = [o for o in obs if o.clinical_feasibility != "INSUFFICIENT"]
            
            mean_nasal_idx = float(np.mean([o.nasal_moisture_index for o in obs]))
            mean_eye_idx = float(np.mean([o.ocular_discharge_index for o in obs]))
            max_nasal_score = max([o.wisconsin_nasal_score for o in obs])
            max_eye_score = max([o.wisconsin_eye_score for o in obs])

            # Cameras observing this cow in this window
            cams = sorted(list(set(o.camera_id for o in obs)))
            # Best camera is Cam 2 if present and feasible
            best_cam = "cam2" if any(o.camera_id == "cam2" and o.clinical_feasibility != "INSUFFICIENT" for o in obs) else cams[0]

            records.append({
                "window_index": w_idx,
                "cow_id": cow_id,
                "window_start_sec": t_start,
                "window_end_sec": t_end,
                "clip_reference": f"{best_cam}_{cow_id}_clip_{int(t_start):05d}_{int(t_end):05d}.mp4",
                "total_frames_sampled": len(obs),
                "feasible_inspection_frames": len(feasible_obs),
                "optical_inspection_feasibility": "FEASIBLE" if len(feasible_obs) > 0 else "INSUFFICIENT",
                "primary_clinical_camera": best_cam,
                "cameras_in_window": cams,
                "mean_nasal_moisture_index": round(mean_nasal_idx, 3),
                "mean_ocular_discharge_index": round(mean_eye_idx, 3),
                "wisconsin_nasal_score_max": max_nasal_score,
                "wisconsin_eye_score_max": max_eye_score,
                "brd_clinical_sign_detected": (max_nasal_score >= 2 or max_eye_score >= 2),
            })
        return records

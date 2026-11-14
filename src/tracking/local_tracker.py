"""Within-camera multi-object tracking using ByteTrack with Kalman filtering."""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    DELETED = 4


class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes in image space [cx, cy, aspect_ratio, height]."""

    count = 0

    def __init__(self, bbox: Tuple[float, float, float, float]):
        # State vector: [cx, cy, s, h, vx, vy, vs, vh]
        # s = area (w * h)
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = w * h
        r = w / float(h)

        self.x = np.array([cx, cy, s, h, 0, 0, 0, 0], dtype=np.float32)

        # State transition matrix
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariances
        self.P = np.diag([10.0, 10.0, 100.0, 10.0, 100.0, 100.0, 1000.0, 100.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 10.0, 1.0, 0.01, 0.01, 0.1, 0.01]).astype(np.float32)
        self.R = np.diag([1.0, 1.0, 10.0, 1.0]).astype(np.float32)

    def predict(self) -> np.ndarray:
        """Advances state vector and returns predicted [x1, y1, x2, y2]."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.get_bbox()

    def update(self, bbox: Tuple[float, float, float, float]):
        """Updates state with observed bounding box."""
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = w * h

        z = np.array([cx, cy, s, h], dtype=np.float32)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Returns [x1, y1, x2, y2] from state."""
        cx, cy, s, h = self.x[0], self.x[1], self.x[2], self.x[3]
        w = max(1.0, s / max(1.0, h))
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return (float(x1), float(y1), float(x2), float(y2))


@dataclass
class LocalTrack:
    track_id: int
    camera_id: str
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    kalman: Optional[KalmanBoxTracker] = None
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    detection_confidence: float = 0.0
    tracking_confidence: float = 0.0
    history: List[Tuple[float, float]] = field(default_factory=list) # List of (cx, y2)
    world_pos: Optional[Tuple[float, float]] = None # (world_x, world_y) in meters
    global_cow_id: Optional[str] = None
    last_frame_number: int = 0
    last_timestamp_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "state": self.state.name,
            "hits": self.hits,
            "age": self.age,
            "time_since_update": self.time_since_update,
            "bbox": [round(c, 1) for c in self.bbox],
            "detection_confidence": round(self.detection_confidence, 2),
            "tracking_confidence": round(self.tracking_confidence, 2),
            "world_pos": [round(c, 2) for c in self.world_pos] if self.world_pos else None,
            "global_cow_id": self.global_cow_id,
        }


def bbox_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0

    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class ByteTracker:
    """ByteTrack implementation with two-stage association for robust occlusion recovery."""

    def __init__(
        self,
        camera_id: str,
        high_det_thresh: float = 0.40,
        low_det_thresh: float = 0.15,
        # Minimum IoU to associate (ByteTrack's 0.8 / 0.5 defaults are 1-IoU *distance* caps,
        # i.e. IoU >= 0.2 for high-score and >= 0.5 for low-score detections). At 1 fps
        # sampling a walking steer moves ~1 m between frames, so IoU 0.7 fragments tracks.
        match_thresh_stage1: float = 0.20,
        match_thresh_stage2: float = 0.50,
        max_time_lost: int = 45,
        min_hits: int = 1,
    ):
        self.camera_id = camera_id
        self.high_det_thresh = high_det_thresh
        self.low_det_thresh = low_det_thresh
        self.match_thresh_stage1 = match_thresh_stage1
        self.match_thresh_stage2 = match_thresh_stage2
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits

        self.tracked_tracks: List[LocalTrack] = []
        self.lost_tracks: List[LocalTrack] = []
        self.next_id = 1
        self.frame_count = 0

    def reset(self):
        """Drop all tracks (e.g. after a hole in the footage) but keep the id counter, so a
        new track never reuses an old (camera, track_id) key known to the identity manager."""
        self.tracked_tracks = []
        self.lost_tracks = []

    def update(
        self,
        detections: List[Any], # List of CowDetection
        frame_number: int,
        timestamp_sec: float,
    ) -> List[LocalTrack]:
        """Runs two-stage ByteTrack update with active and low-confidence detections."""
        self.frame_count += 1

        # 1. Predict new locations of existing tracks using Kalman filter
        for t in self.tracked_tracks + self.lost_tracks:
            t.age += 1
            t.time_since_update += 1  # reset to 0 below when matched this frame
            if t.kalman is not None:
                pred_box = t.kalman.predict()
                t.bbox = pred_box

        # 2. Split detections into high and low confidence pools
        high_dets = [d for d in detections if d.confidence >= self.high_det_thresh]
        low_dets = [
            d for d in detections if self.low_det_thresh <= d.confidence < self.high_det_thresh
        ]

        # 3. First association: Match high_dets with confirmed/active tracks
        active_pool = [t for t in self.tracked_tracks if t.state == TrackState.CONFIRMED]
        unconfirmed_pool = [t for t in self.tracked_tracks if t.state == TrackState.TENTATIVE]
        tracked_pool = active_pool + self.lost_tracks

        matched_tracks_1, unmatched_tracks_1, unmatched_dets_1 = self._associate(
            tracked_pool, high_dets, self.match_thresh_stage1
        )

        for track, det in matched_tracks_1:
            track.kalman.update(det.bbox)
            track.bbox = det.bbox
            track.detection_confidence = det.confidence
            track.tracking_confidence = 0.85 * track.tracking_confidence + 0.15 * det.confidence
            track.hits += 1
            track.time_since_update = 0
            track.state = TrackState.CONFIRMED
            track.last_frame_number = frame_number
            track.last_timestamp_sec = timestamp_sec
            track.history.append(det.ground_point)

        # 4. Second association: Match remaining unmatched tracks with low_dets (recover occluded cows)
        rem_tracks = [tracked_pool[i] for i in unmatched_tracks_1]
        matched_tracks_2, unmatched_tracks_2, _ = self._associate(
            rem_tracks, low_dets, self.match_thresh_stage2
        )

        for track, det in matched_tracks_2:
            track.kalman.update(det.bbox)
            track.bbox = det.bbox
            track.detection_confidence = det.confidence
            track.tracking_confidence = 0.90 * track.tracking_confidence + 0.10 * det.confidence
            track.hits += 1
            track.time_since_update = 0
            track.state = TrackState.CONFIRMED
            track.last_frame_number = frame_number
            track.last_timestamp_sec = timestamp_sec
            track.history.append(det.ground_point)

        # 5. Third association: Match unconfirmed tracks with remaining high_dets
        rem_high_dets = [high_dets[i] for i in unmatched_dets_1]
        matched_unconf, unmatched_unconf, final_unmatched_dets = self._associate(
            unconfirmed_pool, rem_high_dets, self.match_thresh_stage1
        )

        for track, det in matched_unconf:
            track.kalman.update(det.bbox)
            track.bbox = det.bbox
            track.detection_confidence = det.confidence
            track.hits += 1
            track.time_since_update = 0
            if track.hits >= self.min_hits:
                track.state = TrackState.CONFIRMED
            track.last_frame_number = frame_number
            track.last_timestamp_sec = timestamp_sec
            track.history.append(det.ground_point)

        # 6. Initialize new tracks from unassigned high-confidence detections
        for det_idx in final_unmatched_dets:
            det = rem_high_dets[det_idx]
            new_track = LocalTrack(
                track_id=self.next_id,
                camera_id=self.camera_id,
                state=TrackState.CONFIRMED if self.min_hits <= 1 else TrackState.TENTATIVE,
                hits=1,
                age=1,
                time_since_update=0,
                kalman=KalmanBoxTracker(det.bbox),
                bbox=det.bbox,
                detection_confidence=det.confidence,
                tracking_confidence=det.confidence,
                history=[det.ground_point],
                last_frame_number=frame_number,
                last_timestamp_sec=timestamp_sec,
            )
            self.next_id += 1
            self.tracked_tracks.append(new_track)

        # 7. Update track lifecycle (mark Lost or Deleted)
        all_tracks = self.tracked_tracks + self.lost_tracks
        new_tracked: List[LocalTrack] = []
        new_lost: List[LocalTrack] = []

        # Unique tracks
        seen_ids = set()
        for t in all_tracks:
            if t.track_id in seen_ids:
                continue
            seen_ids.add(t.track_id)

            if t.time_since_update == 0:
                new_tracked.append(t)
            elif t.time_since_update <= self.max_time_lost:
                t.state = TrackState.LOST
                new_lost.append(t)
            else:
                t.state = TrackState.DELETED

        self.tracked_tracks = new_tracked
        self.lost_tracks = new_lost

        # Return confirmed active tracks
        return [t for t in self.tracked_tracks if t.state == TrackState.CONFIRMED]

    def _associate(
        self, tracks: List[LocalTrack], detections: List[Any], iou_thresh: float
    ) -> Tuple[List[Tuple[LocalTrack, Any]], List[int], List[int]]:
        """Hungarian linear sum assignment on IoU cost matrix."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                iou = bbox_iou(t.bbox, d.bbox)
                cost_matrix[i, j] = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: List[Tuple[LocalTrack, Any]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= (1.0 - iou_thresh):
                matched.append((tracks[r], detections[c]))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched, sorted(list(unmatched_tracks)), sorted(list(unmatched_dets))

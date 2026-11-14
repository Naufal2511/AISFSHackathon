"""Cross-camera identity association and Global Cow Registry management."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .appearance_gallery import AppearanceGalleryManager
from .local_tracker import LocalTrack
from .consensus_protocol import LyapunovConsensusProtocol
from ..calibration.coordinate_transform import CoordinateTransformer


@dataclass
class GlobalCowRecord:
    global_cow_id: str
    camera_track_ids: Dict[str, List[int]] = field(default_factory=dict) # cam_id -> [local_ids]
    first_seen_timestamp: float = float("inf")
    last_seen_timestamp: float = 0.0
    total_observed_seconds: float = 0.0
    verification_status: str = "PROVISIONAL" # 'PROVISIONAL', 'VERIFIED', 'MANUAL'
    identity_confidence: float = 0.50
    positions: List[Tuple[float, float, float]] = field(default_factory=list) # (time, X, Y)
    cameras_observed: set = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_cow_id": self.global_cow_id,
            "camera_track_ids": {k: list(v) for k, v in self.camera_track_ids.items()},
            "first_seen_timestamp": round(self.first_seen_timestamp, 2)
            if self.first_seen_timestamp != float("inf")
            else None,
            "last_seen_timestamp": round(self.last_seen_timestamp, 2)
            if self.last_seen_timestamp > 0
            else None,
            "total_observed_seconds": round(self.total_observed_seconds, 2),
            "verification_status": self.verification_status,
            "identity_confidence": round(self.identity_confidence, 2),
            "cameras_observed": list(self.cameras_observed),
        }


class GlobalIdentityManager:
    """Associates local camera tracks across multi-camera viewpoints into persistent global IDs."""

    def __init__(
        self,
        transformer: CoordinateTransformer,
        gallery: Optional[AppearanceGalleryManager] = None,
        spatial_distance_threshold_m: float = 3.5,
        appearance_similarity_min: float = 0.60,
        temporal_window_sec: float = 1.0,
    ):
        self.transformer = transformer
        self.gallery = gallery or AppearanceGalleryManager()
        self.spatial_distance_thresh = spatial_distance_threshold_m
        self.appearance_sim_min = appearance_similarity_min
        self.temporal_window = temporal_window_sec

        # 10 cow registry
        self.registry: Dict[str, GlobalCowRecord] = {
            f"COW_{i:02d}": GlobalCowRecord(global_cow_id=f"COW_{i:02d}")
            for i in range(1, 11)
        }

        # Mapping: (camera_id, local_track_id) -> global_cow_id
        self.track_to_global: Dict[Tuple[str, int], str] = {}
        self.assigned_count = 0

        # Lyapunov-stable inter-camera consensus protocol
        self.consensus = LyapunovConsensusProtocol(num_cows=10)
        self.last_consensus_energy: float = 0.0

    def associate_tracks(
        self,
        camera_tracks: Dict[str, List[LocalTrack]],
        track_embeddings: Dict[Tuple[str, int], np.ndarray],
        current_timestamp: float,
    ) -> Dict[Tuple[str, int], str]:
        """
        Cross-camera identity matching algorithm:
        1. Checks if local track already has an assigned global identity.
        2. Compares physical metric positions (world_x, world_y) between cameras.
        3. Compares appearance embeddings using cosine similarity.
        4. Fuses spatial + visual evidence to assign global cow ID or retain provisional status.
        """
        # Step 1: Compute metric ground positions for all active tracks
        active_candidates: List[Dict[str, Any]] = []
        for cam_id, tracks in camera_tracks.items():
            for t in tracks:
                key = (cam_id, t.track_id)
                # Compute ground contact world pos
                cx = (t.bbox[0] + t.bbox[2]) / 2.0
                bottom_y = t.bbox[3]
                world_pos = self.transformer.pixel_to_world(cam_id, cx, bottom_y)
                t.world_pos = world_pos

                emb = track_embeddings.get(key)
                existing_global = self.track_to_global.get(key)

                active_candidates.append(
                    {
                        "camera_id": cam_id,
                        "track": t,
                        "key": key,
                        "world_pos": world_pos,
                        "embedding": emb,
                        "global_id": existing_global,
                    }
                )

        # Step 2: Associate unassigned tracks with active global IDs or across cameras
        unassigned = [c for c in active_candidates if c["global_id"] is None]

        for cand in unassigned:
            cam_a = cand["camera_id"]
            track_a = cand["track"]
            key_a = cand["key"]
            pos_a = cand["world_pos"]
            emb_a = cand["embedding"]

            best_global_id = None
            best_match_score = 0.0

            # Match against currently assigned candidates from other cameras
            for assigned_cand in active_candidates:
                if assigned_cand["global_id"] is None:
                    continue
                if assigned_cand["camera_id"] == cam_a:
                    continue  # Only cross-camera

                pos_b = assigned_cand["world_pos"]
                emb_b = assigned_cand["embedding"]
                target_global = assigned_cand["global_id"]

                # Spatial distance score
                spatial_score = 0.0
                if pos_a is not None and pos_b is not None:
                    dist = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
                    if dist <= self.spatial_distance_thresh:
                        spatial_score = max(0.0, 1.0 - (dist / self.spatial_distance_thresh))

                # Appearance score
                app_score = 0.0
                if emb_a is not None and emb_b is not None:
                    sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-6))
                    if sim >= self.appearance_sim_min:
                        app_score = sim

                # Fused confidence score
                if pos_a is not None and pos_b is not None and emb_a is not None:
                    fused_score = 0.55 * spatial_score + 0.45 * app_score
                elif pos_a is not None and pos_b is not None:
                    fused_score = spatial_score
                elif emb_a is not None and emb_b is not None:
                    fused_score = app_score
                else:
                    fused_score = 0.0

                if fused_score > 0.50 and fused_score > best_match_score:
                    best_match_score = fused_score
                    best_global_id = target_global

            # If confident match found
            if best_global_id is not None:
                self.bind_identity(
                    cam_a, track_a.track_id, best_global_id, confidence=best_match_score
                )
                cand["global_id"] = best_global_id
                track_a.global_cow_id = best_global_id
            else:
                # Query gallery profile if embedding exists
                gallery_cow = None
                sim = 0.0
                if emb_a is not None:
                    gallery_cow, sim = self.gallery.query_best_match(
                        emb_a, min_similarity=self.appearance_sim_min
                    )

                if gallery_cow is not None:
                    self.bind_identity(
                        cam_a, track_a.track_id, gallery_cow, confidence=sim
                    )
                    cand["global_id"] = gallery_cow
                    track_a.global_cow_id = gallery_cow
                else:
                    # Allocate next available COW_XX slot if slots remain
                    if self.assigned_count < 10:
                        self.assigned_count += 1
                        new_id = f"COW_{self.assigned_count:02d}"
                        self.bind_identity(cam_a, track_a.track_id, new_id, confidence=0.55)
                        cand["global_id"] = new_id
                        track_a.global_cow_id = new_id

        # Step 3: Distributed Lyapunov Consensus Agreement across Camera Views
        spatial_matches: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
        local_evidence: Dict[Tuple[str, int], np.ndarray] = {}

        for i, c_a in enumerate(active_candidates):
            gid_a = c_a["global_id"]
            if gid_a and gid_a.startswith("COW_"):
                try:
                    cow_idx = int(gid_a.split("_")[-1]) - 1
                    if 0 <= cow_idx < 10:
                        ev = np.ones(10, dtype=np.float32) * 0.02
                        ev[cow_idx] = 0.82
                        local_evidence[c_a["key"]] = ev
                except ValueError:
                    pass

            for j in range(i + 1, len(active_candidates)):
                c_b = active_candidates[j]
                if c_a["camera_id"] != c_b["camera_id"]:
                    same_id = (gid_a is not None and gid_a == c_b["global_id"])
                    close_dist = False
                    if c_a["world_pos"] is not None and c_b["world_pos"] is not None:
                        d = float(np.linalg.norm(np.array(c_a["world_pos"]) - np.array(c_b["world_pos"])))
                        close_dist = (d <= self.spatial_distance_thresh)
                    if same_id or close_dist:
                        spatial_matches.append((c_a["key"], c_b["key"]))

        if spatial_matches:
            agreed_ids, energy = self.consensus.update_consensus(local_evidence, spatial_matches)
            self.last_consensus_energy = energy
            for key, agreed_cow in agreed_ids.items():
                for c in active_candidates:
                    if c["key"] == key and c["global_id"] != agreed_cow:
                        self.bind_identity(key[0], key[1], agreed_cow, confidence=0.90)
                        c["global_id"] = agreed_cow
                        c["track"].global_cow_id = agreed_cow
        else:
            self.last_consensus_energy = 0.0

        # Step 4: Update registry statistics for observed tracks
        for cand in active_candidates:
            gid = cand["global_id"]
            if gid and gid in self.registry:
                rec = self.registry[gid]
                rec.first_seen_timestamp = min(rec.first_seen_timestamp, current_timestamp)
                rec.last_seen_timestamp = max(rec.last_seen_timestamp, current_timestamp)
                rec.cameras_observed.add(cand["camera_id"])
                if cand["world_pos"]:
                    rec.positions.append((current_timestamp, cand["world_pos"][0], cand["world_pos"][1]))
                # Update gallery exemplar
                if cand["embedding"] is not None:
                    prof = self.gallery.get_profile(gid)
                    if prof:
                        prof.add_exemplar(cand["embedding"], camera_id=cand["camera_id"])

        return self.track_to_global

    def bind_identity(
        self,
        camera_id: str,
        track_id: int,
        global_cow_id: str,
        confidence: float = 0.80,
        status: str = "PROVISIONAL",
    ):
        """Binds a camera track to a global cow registry entry."""
        key = (camera_id, track_id)
        self.track_to_global[key] = global_cow_id
        if global_cow_id in self.registry:
            rec = self.registry[global_cow_id]
            if camera_id not in rec.camera_track_ids:
                rec.camera_track_ids[camera_id] = []
            if track_id not in rec.camera_track_ids[camera_id]:
                rec.camera_track_ids[camera_id].append(track_id)
            rec.identity_confidence = max(rec.identity_confidence, confidence)
            if status == "MANUAL" or rec.verification_status != "MANUAL":
                rec.verification_status = status

    def get_summary(self) -> List[Dict[str, Any]]:
        """Returns structured summary for all 10 cows."""
        summary = []
        for cow_id, rec in self.registry.items():
            duration = (
                max(0.0, rec.last_seen_timestamp - rec.first_seen_timestamp)
                if rec.first_seen_timestamp != float("inf")
                else 0.0
            )
            rec.total_observed_seconds = duration
            summary.append(rec.to_dict())
        return summary

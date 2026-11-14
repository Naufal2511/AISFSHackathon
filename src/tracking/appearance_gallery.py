"""Multi-view appearance gallery for individual cow visual identification."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class CowAppearanceProfile:
    global_cow_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    mean_embedding: Optional[np.ndarray] = None
    exemplar_images: List[np.ndarray] = field(default_factory=list) # Small BGR crops
    view_sources: List[str] = field(default_factory=list) # e.g. ['cam1', 'cam3']
    max_exemplars: int = 10

    def add_exemplar(
        self,
        embedding: np.ndarray,
        image_crop: Optional[np.ndarray] = None,
        camera_id: str = "unknown",
        momentum: float = 0.85,
    ):
        """Adds a verified observation to the gallery and updates running average."""
        if embedding is None or np.all(embedding == 0):
            return

        self.embeddings.append(embedding)
        if len(self.embeddings) > self.max_exemplars:
            self.embeddings.pop(0)

        if image_crop is not None:
            self.exemplar_images.append(image_crop)
            if len(self.exemplar_images) > self.max_exemplars:
                self.exemplar_images.pop(0)

        self.view_sources.append(camera_id)
        if len(self.view_sources) > self.max_exemplars:
            self.view_sources.pop(0)

        # Update running mean embedding with L2 normalization
        if self.mean_embedding is None:
            self.mean_embedding = embedding.copy()
        else:
            self.mean_embedding = momentum * self.mean_embedding + (1.0 - momentum) * embedding
            norm = np.linalg.norm(self.mean_embedding)
            if norm > 1e-6:
                self.mean_embedding = self.mean_embedding / norm

    def cosine_similarity(self, query_embedding: np.ndarray) -> float:
        """Computes cosine similarity between query and gallery mean."""
        if self.mean_embedding is None or query_embedding is None:
            return 0.0
        norm_q = np.linalg.norm(query_embedding)
        norm_m = np.linalg.norm(self.mean_embedding)
        if norm_q < 1e-6 or norm_m < 1e-6:
            return 0.0
        return float(np.dot(self.mean_embedding, query_embedding) / (norm_m * norm_q))


class AppearanceGalleryManager:
    """Manages visual identity galleries across the 10 Angus cows."""

    def __init__(self, expected_cows: int = 10, prefix: str = "COW_"):
        self.profiles: Dict[str, CowAppearanceProfile] = {}
        for i in range(1, expected_cows + 1):
            cow_id = f"{prefix}{i:02d}"
            self.profiles[cow_id] = CowAppearanceProfile(global_cow_id=cow_id)

    def get_profile(self, global_cow_id: str) -> Optional[CowAppearanceProfile]:
        return self.profiles.get(global_cow_id)

    def query_best_match(
        self, query_embedding: np.ndarray, min_similarity: float = 0.60
    ) -> Tuple[Optional[str], float]:
        """Finds closest matching cow ID in the gallery."""
        best_cow_id = None
        best_sim = -1.0

        for cow_id, profile in self.profiles.items():
            if profile.mean_embedding is not None:
                sim = profile.cosine_similarity(query_embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_cow_id = cow_id

        if best_sim >= min_similarity:
            return best_cow_id, best_sim
        return None, best_sim

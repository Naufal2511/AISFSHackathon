"""Visual appearance feature extraction using DINOv2 or ResNet50."""

import torch
import torchvision.transforms as T
import numpy as np
import cv2
from typing import Optional, Tuple


class CowAppearanceExtractor:
    """Extracts invariant visual appearance embeddings from cropped cow regions."""

    def __init__(self, model_name: str = "dinov2_vits14", device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            from ..device import pick_device
            self.device = torch.device(pick_device())
        else:
            self.device = torch.device(device)

        self._init_model()

        # Image transforms
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _init_model(self):
        try:
            if "dinov2" in self.model_name:
                self.model = torch.hub.load(
                    "facebookresearch/dinov2", self.model_name, pretrained=True
                )
                self.embedding_dim = 384
            else:
                import torchvision.models as models

                r50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                # Remove classification head
                self.model = torch.nn.Sequential(*list(r50.children())[:-1])
                self.embedding_dim = 2048

            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            # Fallback to simple spatial color-texture descriptor if offline
            print(f"Notice: Using lightweight fallback embedding extractor ({e})")
            self.model = None
            self.embedding_dim = 128

    def extract(self, frame_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Extracts normalized L2 embedding from cropped cow bbox."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(max(0, c)) for c in bbox]
        x2 = min(w, x2)
        y2 = min(h, y2)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        crop = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if self.model is not None:
            with torch.no_grad():
                tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
                feat = self.model(tensor)
                feat = feat.flatten().cpu().numpy()
                norm = np.linalg.norm(feat)
                if norm > 1e-6:
                    feat = feat / norm
                return feat.astype(np.float32)
        else:
            # Fallback color-spatial histogram
            resized = cv2.resize(crop_rgb, (32, 32))
            hist = cv2.calcHist([resized], [0, 1, 2], None, [4, 4, 8], [0, 256, 0, 256, 0, 256]).flatten()
            norm = np.linalg.norm(hist)
            if norm > 1e-6:
                hist = hist / norm
            return hist.astype(np.float32)

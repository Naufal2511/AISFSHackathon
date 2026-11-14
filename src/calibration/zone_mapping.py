"""Functional pen zone mapping for behavior recognition support."""

import os
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import yaml


class ZoneMapper:
    """Classifies world physical coordinates into pen functional zones."""

    def __init__(self, config_path: str = "configs/pen_layout.yaml"):
        self.config_path = config_path
        self.zones: Dict[str, np.ndarray] = {}
        self.zone_names: Dict[str, str] = {}
        self.load_zones()

    def load_zones(self):
        """Loads metric zone polygons from pen_layout.yaml."""
        if not os.path.exists(self.config_path):
            return

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        zones_data = data.get("zones", {})
        for zone_id, info in zones_data.items():
            pts = info.get("polygon_metric", [])
            if pts:
                self.zones[zone_id] = np.array(pts, dtype=np.float32)
                self.zone_names[zone_id] = info.get("name", zone_id)

    def get_zone(self, world_x: Optional[float], world_y: Optional[float]) -> str:
        """Determines which zone a physical coordinate belongs to."""
        if world_x is None or world_y is None:
            return "UNKNOWN"

        pt = (float(world_x), float(world_y))

        # Check specific functional zones first
        for zone_id in ["feeding_zone", "drinking_zone", "resting_zone"]:
            if zone_id in self.zones:
                poly = self.zones[zone_id]
                res = cv2.pointPolygonTest(poly, pt, measureDist=False)
                if res >= 0:
                    return self.zone_names.get(zone_id, zone_id)

        # Fallback to general movement zone or inside pen
        if "movement_zone" in self.zones:
            poly = self.zones["movement_zone"]
            if cv2.pointPolygonTest(poly, pt, measureDist=False) >= 0:
                return self.zone_names.get("movement_zone", "General Movement")

        # If inside pen rectangle
        if 0.0 <= world_x <= 12.0 and 0.0 <= world_y <= 30.0:
            return "General Pen Area"

        return "Outside / Unmapped"

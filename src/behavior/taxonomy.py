"""The 10 official hackathon behavior classes and label maps for public datasets."""

from typing import Dict, List

BEHAVIORS: List[str] = [
    "lying",
    "standing",
    "walking",
    "feeding",
    "drinking",
    "ruminating (lying)",
    "ruminating (standing)",
    "grooming",
    "social interaction",
    "other / not visible",
]
IDX: Dict[str, int] = {b: i for i, b in enumerate(BEHAVIORS)}
NOT_VISIBLE = "other / not visible"

# Behaviors that imply the animal is lying (used by posture-consistency checks).
LYING_FAMILY = {"lying", "ruminating (lying)"}

# CVB (CSIRO, arXiv:2305.16555): 8 Angus on pasture, 11 classes. Licence CC BY-NC-SA 4.0.
CVB_TO_OURS: Dict[str, str] = {
    "grazing": "feeding",
    "walking": "walking",
    "running": "walking",
    "ruminating-standing": "ruminating (standing)",
    "ruminating-lying": "ruminating (lying)",
    "resting-standing": "standing",
    "resting-lying": "lying",
    "drinking": "drinking",
    "grooming": "grooming",
    "other": NOT_VISIBLE,
    "hidden": NOT_VISIBLE,
}

# CBVD-5 (Kaggle fandaoerji/cbvd-5cow-behavior-video-dataset), CC0. Rumination is not
# split by posture there, so it is resolved with the posture model at train time.
CBVD5_TO_OURS: Dict[str, str] = {
    "standing": "standing",
    "lying down": "lying",
    "lying": "lying",
    "feeding": "feeding",
    "foraging": "feeding",
    "drinking": "drinking",
    "drinking water": "drinking",
    "rumination": "ruminating",  # resolved to (lying)/(standing) via posture
    "ruminating": "ruminating",
}
CBVD5_ID_TO_NAME = {0: "standing", 1: "lying down", 2: "feeding", 3: "drinking", 4: "rumination"}

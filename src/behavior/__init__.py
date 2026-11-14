"""Level-1 behavior classification (10 official classes) on top of multi-camera tracking."""
from .taxonomy import BEHAVIORS
from .classifier import BehaviorClassifier

__all__ = ["BEHAVIORS", "BehaviorClassifier"]

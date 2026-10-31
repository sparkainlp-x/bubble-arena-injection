"""bubble-arena-injection: arena + Crystallize injection under a bubble split."""

from .bubble import BubbleArena
from .calibrator import Calibrator, FailureCapsule
from .observer import BubbleObserver, Report, half_width

__all__ = ["BubbleArena", "BubbleObserver", "Calibrator", "FailureCapsule", "Report", "half_width"]
__version__ = "0.1.0"

"""
Protocols and Data Structures for Pluggable External Routines.
Allows researchers to define custom clustering, decision-making, and live feedback
algorithms in separate Python scripts or notebooks.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from ..imaging.clustering import Cluster


class DecisionAction(str, Enum):
    ACQUIRE = "ACQUIRE"             # Proceed with standard acquisition
    SKIP = "SKIP"                   # Skip this target (e.g. empty vacuum or carbon film)
    ADJUST_DWELL = "ADJUST_DWELL"   # Dynamically alter exposure / dwell time
    ABORT = "ABORT"                 # Stop execution immediately (e.g. sample burned/damaged)


@dataclass
class SurveyData:
    """Survey image and associated column calibration data passed to decision routines."""
    image: np.ndarray
    stage_x_nm: float
    stage_y_nm: float
    nm_per_pixel: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanTarget:
    """Represents a targeted region or point decided by an algorithm."""
    target_id: int
    x_px: int
    y_px: int
    width_px: int
    height_px: int
    stage_x_nm: float
    stage_y_nm: float
    dwell_time_s: float = 1e-3
    priority: int = 1
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveDecision:
    """Feedback action returned by an in-flight live analysis routine."""
    action: DecisionAction = DecisionAction.ACQUIRE
    new_dwell_s: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type aliases for external pluggable functions
FilterFunc = Callable[[np.ndarray, Dict[str, Any]], np.ndarray]
ClusteringFunc = Callable[[np.ndarray, Dict[str, Any]], List[Cluster]]
DecisionFunc = Callable[[SurveyData, Dict[str, Any]], List[ScanTarget]]
LiveFeedbackFunc = Callable[[np.ndarray, ScanTarget, Dict[str, Any]], LiveDecision]

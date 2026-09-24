"""
Base classes and protocols for pluggable in-flight microscopy corrections.
Supports independent drift, autofocus, and future custom corrections with
configurable scan intervals and reference location strategies (previous scan,
next scan, or a fixed cluster region).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from ..hardware.base import BaseMicroscope, BaseScanner
from ..imaging.grid_planner import ScanTile
from ..imaging.clustering import Cluster


class CorrectionLocationMode(str, Enum):
    """Specifies where the microscope points to evaluate the correction."""
    PREVIOUS_SCAN = "PREVIOUS_SCAN"        # Use preceding scan tile
    NEXT_SCAN = "NEXT_SCAN"                # Evaluate at upcoming scan tile target
    FIXED_REGION = "FIXED_REGION"          # Return to a designated reference cluster/ROI
    ARBITRARY_POSITION = "ARBITRARY_POSITION" # Return to an arbitrary user-chosen coordinate (stage X/Y or survey pixel)


@dataclass
class CorrectionContext:
    """Context passed to each correction routine before a 4D-STEM scan."""
    scan_index: int                       # 0-indexed count of completed scans
    total_scans: int                      # Total number of planned scans
    current_tile: ScanTile                # Upcoming scan tile to be collected
    previous_tile: Optional[ScanTile]     # Preceding completed scan tile
    fixed_cluster: Optional[Cluster]      # Fixed anchor cluster if FIXED_REGION is used
    arbitrary_position_nm: Optional[Tuple[float, float]] = None # Arbitrary stage coordinate (x, y)
    arbitrary_position_px: Optional[Tuple[int, int]] = None     # Arbitrary pixel coordinate in survey
    microscope: BaseMicroscope = None     # Microscope interface
    scanner: BaseScanner = None           # Scanner interface
    survey_image: Optional[np.ndarray] = None # Original survey image
    nm_per_pixel: float = 1.0


@dataclass
class CorrectionResult:
    """Outcome of a correction execution."""
    routine_name: str
    executed: bool
    dx_nm: float = 0.0
    dy_nm: float = 0.0
    defocus_delta: float = 0.0
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseCorrectionRoutine(ABC):
    """
    Abstract Base Class for an in-flight correction routine.
    Any new correction (beam tilt, stigmation, thermal drift, etc.)
    can be implemented by subclassing this class.
    """

    def __init__(
        self,
        name: str,
        enabled: bool = False,
        interval_scans: int = 5,
        location_mode: CorrectionLocationMode = CorrectionLocationMode.FIXED_REGION,
        fixed_cluster_id: Optional[int] = None,
        arbitrary_position_nm: Optional[Tuple[float, float]] = None,
        arbitrary_position_px: Optional[Tuple[int, int]] = None
    ):
        self.name = name
        self.enabled = enabled
        self.interval_scans = max(1, int(interval_scans))
        self.location_mode = location_mode
        self.fixed_cluster_id = fixed_cluster_id
        self.arbitrary_position_nm = arbitrary_position_nm
        self.arbitrary_position_px = arbitrary_position_px
        self.last_run_scan_index: int = -1

    def should_run(self, context: CorrectionContext) -> bool:
        """
        Determine if this correction should trigger before the current scan.
        Triggers every interval_scans 4D-STEM scans.
        """
        if not self.enabled:
            return False

        # First scan trigger check (optional: run at scan 0 if configured)
        if context.scan_index == 0:
            return False

        # Check if interval has elapsed since last run or based on scan_index
        return (context.scan_index % self.interval_scans) == 0

    @abstractmethod
    def execute(self, context: CorrectionContext) -> CorrectionResult:
        """
        Execute the correction routine.
        Must adjust microscope parameters if necessary and return a CorrectionResult.
        """
        pass

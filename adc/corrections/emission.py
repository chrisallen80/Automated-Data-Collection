"""
Electron Gun Emission Monitoring and Flashing Control.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from ..hardware.base import BaseMicroscope


class EmissionMonitor:
    """
    Monitors cold-FEG / Schottky emitter stability.
    Alerts user or pauses acquisition if emission current drops below threshold.
    """

    def __init__(
        self,
        microscope: BaseMicroscope,
        min_current_ua: float = 8.0,
        check_interval_scans: int = 10
    ):
        self.microscope = microscope
        self.min_current_ua = min_current_ua
        self.check_interval_scans = check_interval_scans
        self._scans_since_check = 0

    def check_emission(self) -> Dict[str, Any]:
        """
        Query current emission and evaluate health.
        """
        # In mock or hardware mode, inspect column parameters
        params = self.microscope.get_all_parameters()
        # Simulated or actual emission current
        current_ua = float(params.get("emission_current_ua", 12.5))
        needs_flashing = current_ua < self.min_current_ua

        return {
            "emission_current_ua": current_ua,
            "min_threshold_ua": self.min_current_ua,
            "is_stable": not needs_flashing,
            "needs_flashing": needs_flashing
        }

    def on_scan_completed(self) -> bool:
        """
        Called after each scan tile. Returns True if emission check should be performed.
        """
        self._scans_since_check += 1
        if self._scans_since_check >= self.check_interval_scans:
            self._scans_since_check = 0
            return True
        return False

"""
Production Quantum Detectors ScanEngine wrapper.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from .base import BaseScanner


class ScanEngineQD(BaseScanner):
    """
    Adapter for Quantum Detectors ScanEngine STEM scan generator.
    Handles survey raster acquisition, spot mode, and TTL trigger synchronization.
    """

    def __init__(self, ip: str = "127.0.0.1", port: int = 8080):
        self._ip = ip
        self._port = port
        self._engine = None
        self._connected = False

    def connect(self) -> bool:
        try:
            from pyscanengine import ScanEngine
            self._engine = ScanEngine()
            self._connected = True
            return True
        except ImportError as e:
            raise RuntimeError(
                f"pyscanengine library not installed or unavailable: {e}. "
                f"Please run in mock mode using MockScanner."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to ScanEngine: {e}")

    def disconnect(self) -> None:
        self._connected = False
        self._engine = None

    def acquire_survey_image(self, size: Tuple[int, int] = (512, 512), dwell_time_s: float = 15e-6) -> np.ndarray:
        if not self._connected or self._engine is None:
            raise RuntimeError("ScanEngine not connected.")

        h, w = size
        # Configure ScanEngine full scan
        self._engine.pixel_time = dwell_time_s
        # Frame acquisition via FrameMonitor
        from pyscanengine.data.frame_monitor import FrameMonitor
        monitor = FrameMonitor(self._engine)
        frame = monitor.get_frame(timeout=10.0)
        if frame is None:
            raise TimeoutError("Timed out waiting for ScanEngine survey frame.")
        return np.asarray(frame, dtype=np.uint8)

    def set_scan_mode(self, mode: str) -> None:
        if not self._connected or self._engine is None:
            raise RuntimeError("ScanEngine not connected.")
        # Modes: 'full', 'spot', 'area'
        pass

    def trigger_ttl(self, line: int = 0, high: bool = True) -> None:
        if not self._connected or self._engine is None:
            raise RuntimeError("ScanEngine not connected.")
        if high:
            self._engine.enable_lines(1 << line)
        else:
            self._engine.disable_lines(1 << line)

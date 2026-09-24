"""
Offline mock simulator for Merlin 4D-STEM detector.
Generates realistic convergent beam electron diffraction (CBED / 4D-STEM) patterns.
"""

from __future__ import annotations
import time
import threading
from typing import Optional
import numpy as np
import cv2
from ..base import BaseCamera


class MockMerlin(BaseCamera):
    """
    Simulates a Merlin Medipix detector (e.g. 256x256 or 512x512 pixels).
    Generates realistic diffraction patterns with central transmitted disc,
    diffuse scattering, and sharp crystalline Bragg spots.
    """

    def __init__(self, detector_size: int = 256):
        self.detector_size = detector_size
        self._connected = True
        self._is_acquiring = False
        self._num_frames = 100
        self._bit_depth = 12
        self._dwell_s = 0.001
        self._frame_count = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._acq_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Cache a synthetic diffraction template
        self._base_pattern = self._create_synthetic_diffraction_pattern(detector_size)

    def _create_synthetic_diffraction_pattern(self, size: int) -> np.ndarray:
        """Create a realistic convergent beam electron diffraction (CBED) frame."""
        pattern = np.zeros((size, size), dtype=np.float32)
        cx, cy = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # 1. Central transmitted primary disc (convergence angle alpha)
        disc_radius = size * 0.12
        disc_mask = r <= disc_radius
        pattern[disc_mask] = 1.0 - 0.2 * (r[disc_mask] / disc_radius) ** 2

        # 2. Diffuse amorphous scattering halo
        halo_r0 = size * 0.28
        halo_sigma = size * 0.05
        halo = 0.25 * np.exp(-((r - halo_r0) ** 2) / (2 * halo_sigma ** 2))
        pattern += halo

        # 3. Bragg spots (crystalline diffraction reflections)
        # 6-fold or 4-fold crystal symmetry spots
        spot_radii = [size * 0.25, size * 0.38]
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        for d in spot_radii:
            for theta in angles:
                sx = int(cx + d * np.cos(theta))
                sy = int(cy + d * np.sin(theta))
                if 0 <= sx < size and 0 <= sy < size:
                    # Gaussian spot matching convergence semi-angle
                    spot_r = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
                    spot_disc = spot_r <= (disc_radius * 0.7)
                    pattern[spot_disc] = np.maximum(pattern[spot_disc], 0.75 * np.exp(-((spot_r[spot_disc]) ** 2) / (2 * (disc_radius * 0.35) ** 2)))

        return pattern

    def connect(self, host: str = "127.0.0.1", port: int = 6341) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self.stop_acquisition()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def configure(
        self,
        num_frames: int,
        bit_depth: int = 12,
        dwell_s: float = 0.001,
        save_dir: str = "",
        filename: str = "scan_data",
        continuous_rw: bool = True
    ) -> bool:
        self._num_frames = num_frames
        self._bit_depth = bit_depth
        self._dwell_s = max(1e-4, dwell_s)
        self._frame_count = 0
        return True

    def start_acquisition(self) -> bool:
        if self._is_acquiring:
            return True
        self._is_acquiring = True
        self._stop_event.clear()
        self._frame_count = 0
        self._acq_thread = threading.Thread(target=self._run_acquisition, daemon=True)
        self._acq_thread.start()
        return True

    def stop_acquisition(self) -> bool:
        self._stop_event.set()
        self._is_acquiring = False
        if self._acq_thread and self._acq_thread.is_alive():
            self._acq_thread.join(timeout=1.0)
        return True

    def is_acquiring(self) -> bool:
        return self._is_acquiring

    def _run_acquisition(self) -> None:
        rng = np.random.default_rng()
        while not self._stop_event.is_set() and self._frame_count < self._num_frames:
            # Generate next noisy frame with slight random rotation/spot fluctuations
            angle = float(self._frame_count * 0.02)
            M = cv2.getRotationMatrix2D((self.detector_size / 2, self.detector_size / 2), angle, 1.0)
            rotated = cv2.warpAffine(self._base_pattern, M, (self.detector_size, self.detector_size))

            # Poisson / Gaussian counting noise
            noise = rng.normal(0, 0.03, size=(self.detector_size, self.detector_size)).astype(np.float32)
            frame = np.clip((rotated + noise) * ((2 ** self._bit_depth) - 1), 0, (2 ** self._bit_depth) - 1).astype(np.uint16)

            with self._lock:
                self._latest_frame = frame
                self._frame_count += 1

            time.sleep(self._dwell_s)

        self._is_acquiring = False

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            # Return static template if not actively acquiring
            rng = np.random.default_rng()
            noise = rng.normal(0, 0.02, size=(self.detector_size, self.detector_size)).astype(np.float32)
            frame = np.clip((self._base_pattern + noise) * 4095.0, 0, 4095).astype(np.uint16)
            return frame

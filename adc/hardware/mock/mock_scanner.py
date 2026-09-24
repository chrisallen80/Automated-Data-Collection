"""
Offline mock simulator for STEM scanning engine.
Generates synthetic STEM images (nanoparticles on amorphous carbon support)
that respond to stage translations and beam blanking.
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Optional
from ..base import BaseScanner
from .mock_microscope import MockMicroscope


class MockScanner(BaseScanner):
    """
    Simulates a STEM scan engine (e.g. Quantum Detectors ScanEngine).
    Renders high-fidelity synthetic STEM survey micrographs.
    """

    def __init__(self, microscope: Optional[MockMicroscope] = None):
        self._connected = True
        self._microscope = microscope
        self._scan_mode = "full"
        self._last_ttl_state = False

        # Pre-generate a large "master specimen" field (2048 x 2048)
        self._master_size = 2048
        self._master_specimen = self._create_synthetic_specimen(self._master_size)

    def _create_synthetic_specimen(self, size: int) -> np.ndarray:
        """Create a synthetic specimen field containing nanoparticles and amorphous texture."""
        rng = np.random.default_rng(42)

        # 1. Amorphous carbon background texture with low-pass filtered noise
        noise = rng.normal(loc=0.3, scale=0.08, size=(size, size)).astype(np.float32)
        bg = cv2.GaussianBlur(noise, (11, 11), 3.0)

        # 2. Add synthetic gold nanoparticles / crystal grains
        field = bg.copy()
        num_particles = 45
        for _ in range(num_particles):
            cx = rng.integers(100, size - 100)
            cy = rng.integers(100, size - 100)
            r = rng.integers(15, 60)
            intensity = rng.uniform(0.6, 0.95)

            # Circular / faceted particle
            y, x = np.ogrid[:size, :size]
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            mask = dist_sq <= r ** 2
            field[mask] = np.maximum(field[mask], intensity)

            # Subtle internal lattice contrast
            lattice_freq = 0.4
            lattice = 0.05 * np.sin(x * lattice_freq) * np.sin(y * lattice_freq)
            field[mask] += lattice[mask]

        # 3. Add a few small clusters/aggregates
        for _ in range(5):
            cx = rng.integers(200, size - 200)
            cy = rng.integers(200, size - 200)
            for _ in range(rng.integers(3, 7)):
                ox = cx + rng.integers(-40, 40)
                oy = cy + rng.integers(-40, 40)
                r = rng.integers(10, 30)
                y, x = np.ogrid[:size, :size]
                mask = ((x - ox) ** 2 + (y - oy) ** 2) <= r ** 2
                field[mask] = np.maximum(field[mask], 0.85)

        return np.clip(field, 0.0, 1.0).astype(np.float32)

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def acquire_survey_image(self, size: Tuple[int, int] = (512, 512), dwell_time_s: float = 15e-6) -> np.ndarray:
        """
        Crop a field of view from master specimen corresponding to current stage coordinates.
        Adds detector shot noise inversely proportional to dwell time.
        """
        h, w = size

        if self._microscope and self._microscope.get_beam_blank():
            # If beam is blanked, return pure dark noise
            rng = np.random.default_rng()
            return (rng.normal(loc=0.01, scale=0.005, size=(h, w)).clip(0, 1) * 255).astype(np.uint8)

        # Coordinate offset from microscope stage (1 nm = 0.5 pixels)
        ox, oy = 0, 0
        if self._microscope:
            pos = self._microscope.get_stage_position()
            # Map nanometers into pixel displacement modulo wrap
            ox = int((pos.x_nm * 0.5)) % (self._master_size - w)
            oy = int((pos.y_nm * 0.5)) % (self._master_size - h)
        else:
            ox = (self._master_size - w) // 2
            oy = (self._master_size - h) // 2

        crop = self._master_specimen[oy:oy + h, ox:ox + w]

        # Defocus blur simulation: if microscope OLf deviates from 38942
        if self._microscope:
            defocus_delta = abs(self._microscope.get_defocus() - 38942)
            if defocus_delta > 10:
                ksize = min(31, int(defocus_delta // 10) * 2 + 1)
                crop = cv2.GaussianBlur(crop, (ksize, ksize), ksize * 0.3)

        # Shot noise based on dwell time
        noise_level = np.clip(1e-5 / (dwell_time_s + 1e-9) * 0.02, 0.005, 0.15)
        rng = np.random.default_rng()
        noisy = crop + rng.normal(0, noise_level, size=(h, w)).astype(np.float32)
        uint8_img = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
        return uint8_img

    def set_scan_mode(self, mode: str) -> None:
        self._scan_mode = mode

    def trigger_ttl(self, line: int = 0, high: bool = True) -> None:
        self._last_ttl_state = high

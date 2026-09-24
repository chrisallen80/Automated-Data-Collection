"""
Offline mock simulator for Energy Dispersive X-ray (EDX) spectrometer.
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, List
import numpy as np
from ..base import BaseEDX


class MockEDX(BaseEDX):
    """
    Simulates an EDX spectrometer. Generates realistic X-ray emission
    spectra with Bremsstrahlung background and characteristic peak lines
    (e.g., C-K, O-K, Si-K, Fe-K, Cu-K, Au-L).
    """

    # Common characteristic X-ray line energies in keV
    PEAKS = {
        "C": 0.277,
        "O": 0.525,
        "Si": 1.740,
        "Fe": 6.404,
        "Cu": 8.046,
        "Au": 9.713,
    }

    def __init__(self, num_channels: int = 2048, max_energy_kev: float = 20.0):
        self._connected = True
        self.num_channels = num_channels
        self.max_energy_kev = max_energy_kev
        self.energies = np.linspace(0, max_energy_kev, num_channels)

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def acquire_point_spectrum(self, dwell_s: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic EDX spectrum with Bremsstrahlung background and peaks.
        Returns (energies_keV, counts).
        """
        rng = np.random.default_rng()

        # 1. Bremsstrahlung background (Kramers' Law approx: I(E) ~ (E_max - E)/E)
        e = np.maximum(self.energies, 0.2)
        background = 120.0 * (self.max_energy_kev - e) / e * np.exp(-0.4 / e)
        background = np.clip(background, 0, None)

        # 2. Add characteristic Gaussian peaks (detector energy resolution ~ 130 eV FWHM)
        sigma = 0.06 # keV (~140 eV FWHM)
        spectrum = background.copy()

        sample_elements = ["C", "O", "Si", "Cu"]
        for elem in sample_elements:
            peak_e = self.PEAKS[elem]
            amplitude = rng.uniform(300, 1500) * dwell_s
            gaussian = amplitude * np.exp(-((self.energies - peak_e) ** 2) / (2 * sigma ** 2))
            spectrum += gaussian

        # 3. Poisson counting statistics
        counts = rng.poisson(np.maximum(spectrum * dwell_s, 0)).astype(np.int32)
        return self.energies, counts

    def acquire_elemental_map(
        self,
        size: Tuple[int, int],
        dwell_s: float = 0.01,
        elements: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Generate synthetic 2D elemental distribution maps."""
        h, w = size
        if elements is None:
            elements = ["Si", "Cu", "O", "C"]

        rng = np.random.default_rng(42)
        maps = {}
        for elem in elements:
            # Generate random smooth spatial gradient / particle distribution
            base = rng.uniform(5, 20, size=(h, w)).astype(np.float32)
            if elem in ("Cu", "Si"):
                # Spotty particle concentration
                cx, cy = w // 2, h // 2
                y, x = np.ogrid[:h, :w]
                r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                base += 80.0 * np.exp(-(r ** 2) / (2 * (min(h, w) * 0.2) ** 2))

            counts = rng.poisson(np.clip(base * (dwell_s / 0.01), 0, None)).astype(np.int32)
            maps[elem] = counts

        return maps

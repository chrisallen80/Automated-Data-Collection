"""
EDX Hardware Adapter.
Provides pluggable interface for Oxford Instruments, Bruker, or JEOL EDX spectrometer scripts.
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, List, Callable
import numpy as np
from .base import BaseEDX


class EDXAdapter(BaseEDX):
    """
    Adapter allowing user custom EDX acquisition scripts / drivers to be plugged in.
    Supports point spectra and elemental mapping.
    """

    def __init__(
        self,
        acquire_spectrum_fn: Optional[Callable[[float], Tuple[np.ndarray, np.ndarray]]] = None,
        acquire_map_fn: Optional[Callable[[Tuple[int, int], float, Optional[List[str]]], Dict[str, np.ndarray]]] = None
    ):
        self._connected = True
        self._custom_spectrum_fn = acquire_spectrum_fn
        self._custom_map_fn = acquire_map_fn

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def acquire_point_spectrum(self, dwell_s: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        if self._custom_spectrum_fn is not None:
            return self._custom_spectrum_fn(dwell_s)
        # Default placeholder if no custom script is bound
        channels = 2048
        energies = np.linspace(0, 20.0, channels)
        counts = np.zeros(channels, dtype=np.int32)
        return energies, counts

    def acquire_elemental_map(
        self,
        size: Tuple[int, int],
        dwell_s: float = 0.01,
        elements: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        if self._custom_map_fn is not None:
            return self._custom_map_fn(size, dwell_s, elements)
        h, w = size
        elems = elements or ["Si", "Cu"]
        return {elem: np.zeros((h, w), dtype=np.int32) for elem in elems}

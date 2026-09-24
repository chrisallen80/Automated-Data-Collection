"""
Hardware Abstraction Layer (HAL) base interfaces.
Provides unified, decoupled abstractions for electron microscope, scanning engine,
4D-STEM detector (Merlin), and EDX spectrometer.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any
import numpy as np


@dataclass
class StagePosition:
    x_nm: float = 0.0
    y_nm: float = 0.0
    z_nm: float = 0.0
    tx_deg: float = 0.0
    ty_deg: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        return (self.x_nm, self.y_nm, self.z_nm, self.tx_deg, self.ty_deg)


@dataclass
class LensState:
    olf: int = 0          # Objective Lens Fine
    olc: int = 0          # Objective Lens Coarse
    cl1: int = 0
    cl2: int = 0
    cl3: int = 0
    mag: float = 10000.0  # Magnification
    spot_size: int = 1
    ht_kv: float = 200.0  # High tension in kV


class BaseMicroscope(ABC):
    """Abstract interface for electron microscope control (e.g. JEOL ARM200F via PyJEM)."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to microscope hardware / service."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect safely from microscope hardware."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is alive."""
        pass

    @abstractmethod
    def get_stage_position(self) -> StagePosition:
        """Return current stage position in nanometers and tilt in degrees."""
        pass

    @abstractmethod
    def set_stage_position(self, x_nm: Optional[float] = None, y_nm: Optional[float] = None) -> StagePosition:
        """Move stage to absolute coordinates."""
        pass

    @abstractmethod
    def set_stage_relative(self, dx_nm: float, dy_nm: float) -> StagePosition:
        """Move stage relatively by dx and dy in nanometers."""
        pass

    @abstractmethod
    def get_defocus(self) -> int:
        """Get Objective Lens Fine (OLf) defocus value."""
        pass

    @abstractmethod
    def set_defocus(self, value: int) -> None:
        """Set Objective Lens Fine (OLf) defocus value."""
        pass

    @abstractmethod
    def get_beam_blank(self) -> bool:
        """Return True if beam is currently blanked."""
        pass

    @abstractmethod
    def set_beam_blank(self, blank: bool) -> None:
        """Enable (True) or disable (False) beam blanking."""
        pass

    @abstractmethod
    def get_lens_state(self) -> LensState:
        """Retrieve snapshot of current lens and column settings."""
        pass

    @abstractmethod
    def get_all_parameters(self) -> Dict[str, Any]:
        """Fetch dictionary of metadata parameters suitable for HDF5 archival."""
        pass


class BaseScanner(ABC):
    """Abstract interface for STEM scan generator (e.g. Quantum Detectors ScanEngine or JEOL Digiscan)."""

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def acquire_survey_image(self, size: Tuple[int, int] = (512, 512), dwell_time_s: float = 15e-6) -> np.ndarray:
        """Acquire a 2D survey STEM image (e.g. ADF / BF). Returns 2D float32 or uint16 numpy array."""
        pass

    @abstractmethod
    def set_scan_mode(self, mode: str) -> None:
        """Set scan mode: 'full', 'spot', or 'area'."""
        pass

    @abstractmethod
    def trigger_ttl(self, line: int = 0, high: bool = True) -> None:
        """Send TTL synchronization pulse to external camera."""
        pass


class BaseCamera(ABC):
    """Abstract interface for 4D-STEM detector (e.g. Merlin Medipix TCP camera)."""

    @abstractmethod
    def connect(self, host: str, port: int) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def configure(
        self,
        num_frames: int,
        bit_depth: int = 12,
        dwell_s: float = 0.001,
        save_dir: str = "",
        filename: str = "scan_data",
        continuous_rw: bool = True
    ) -> bool:
        """Configure camera acquisition parameters."""
        pass

    @abstractmethod
    def start_acquisition(self) -> bool:
        """Start waiting for triggers / acquiring frames."""
        pass

    @abstractmethod
    def stop_acquisition(self) -> bool:
        """Halt active acquisition."""
        pass

    @abstractmethod
    def is_acquiring(self) -> bool:
        """Check whether camera is currently acquiring."""
        pass

    @abstractmethod
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Fetch the most recent 2D diffraction pattern (e.g. 256x256 or 512x512)."""
        pass


class BaseEDX(ABC):
    """Abstract interface for Energy Dispersive X-ray (EDX) spectrometer."""

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def acquire_point_spectrum(self, dwell_s: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acquire EDX spectrum at current probe location.
        Returns (energies_keV, counts).
        """
        pass

    @abstractmethod
    def acquire_elemental_map(
        self,
        size: Tuple[int, int],
        dwell_s: float = 0.01,
        elements: Optional[list[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Acquire 2D elemental distribution map for specified elements.
        Returns dict mapping element symbol -> 2D intensity map.
        """
        pass

"""
Realistic offline mock simulator for JEOL ARM200F / PyJEM.
"""

from __future__ import annotations
import time
from typing import Optional, Dict, Any
from ..base import BaseMicroscope, StagePosition, LensState


class MockMicroscope(BaseMicroscope):
    """
    Simulates electron microscope stage movements, lens excitations,
    and beam blanking for offline development and testing.
    """

    def __init__(self, initial_olf: int = 38942):
        self._connected = True
        self._position = StagePosition(x_nm=10000.0, y_nm=15000.0, z_nm=500.0, tx_deg=0.0, ty_deg=0.0)
        self._olf = initial_olf
        self._olc = 12500
        self._cl1 = 32000
        self._cl2 = 28000
        self._cl3 = 30000
        self._mag = 50000.0
        self._spot_size = 5
        self._ht_kv = 200.0
        self._beam_blank = False
        self._drift_rate_nm_s = (0.05, -0.03) # subtle natural physical drift
        self._last_time = time.time()

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_stage_position(self) -> StagePosition:
        # simulate slight thermal drift over time
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        self._position.x_nm += self._drift_rate_nm_s[0] * dt
        self._position.y_nm += self._drift_rate_nm_s[1] * dt
        return StagePosition(
            x_nm=self._position.x_nm,
            y_nm=self._position.y_nm,
            z_nm=self._position.z_nm,
            tx_deg=self._position.tx_deg,
            ty_deg=self._position.ty_deg,
        )

    def set_stage_position(self, x_nm: Optional[float] = None, y_nm: Optional[float] = None) -> StagePosition:
        if x_nm is not None:
            self._position.x_nm = float(x_nm)
        if y_nm is not None:
            self._position.y_nm = float(y_nm)
        self._last_time = time.time()
        return self.get_stage_position()

    def set_stage_relative(self, dx_nm: float, dy_nm: float) -> StagePosition:
        self._position.x_nm += float(dx_nm)
        self._position.y_nm += float(dy_nm)
        self._last_time = time.time()
        return self.get_stage_position()

    def get_defocus(self) -> int:
        return self._olf

    def set_defocus(self, value: int) -> None:
        self._olf = int(value)

    def get_beam_blank(self) -> bool:
        return self._beam_blank

    def set_beam_blank(self, blank: bool) -> None:
        self._beam_blank = bool(blank)

    def get_lens_state(self) -> LensState:
        return LensState(
            olf=self._olf,
            olc=self._olc,
            cl1=self._cl1,
            cl2=self._cl2,
            cl3=self._cl3,
            mag=self._mag,
            spot_size=self._spot_size,
            ht_kv=self._ht_kv
        )

    def get_all_parameters(self) -> Dict[str, Any]:
        pos = self.get_stage_position()
        return {
            "stage_x_nm": pos.x_nm,
            "stage_y_nm": pos.y_nm,
            "stage_z_nm": pos.z_nm,
            "tilt_x_deg": pos.tx_deg,
            "tilt_y_deg": pos.ty_deg,
            "magnification": self._mag,
            "high_tension_kv": self._ht_kv,
            "spot_size": self._spot_size,
            "objective_lens_fine": self._olf,
            "objective_lens_coarse": self._olc,
            "beam_blank": self._beam_blank,
            "mock_mode": True
        }

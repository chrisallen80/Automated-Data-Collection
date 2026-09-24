"""
Production PyJEM adapter for JEOL ARM200F microscope.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from .base import BaseMicroscope, StagePosition, LensState


class PyJEMAdapter(BaseMicroscope):
    """
    Adapter interfacing directly with JEOL TEM3 subsystem via PyJEM.
    Controls stage, objective lens fine (OLf), beam blanking, and column parameters.
    """

    def __init__(self):
        self._connected = False
        self._stage = None
        self._lens = None
        self._def = None
        self._eos = None
        self._ht = None
        self._apt = None

    def connect(self) -> bool:
        try:
            from PyJEM import TEM3
            self._stage = TEM3.Stage3()
            self._lens = TEM3.Lens3()
            self._def = TEM3.Def3()
            self._eos = TEM3.EOS3()
            self._ht = TEM3.HT3()
            self._apt = TEM3.Apt3()
            self._connected = True
            return True
        except ImportError as e:
            raise RuntimeError(
                f"PyJEM library is not installed or unavailable in this environment: {e}. "
                f"Please run in mock mode using MockMicroscope."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to JEOL microscope via PyJEM: {e}")

    def disconnect(self) -> None:
        self._connected = False
        self._stage = None
        self._lens = None
        self._def = None
        self._eos = None
        self._ht = None
        self._apt = None

    def is_connected(self) -> bool:
        return self._connected

    def get_stage_position(self) -> StagePosition:
        if not self._connected or self._stage is None:
            raise RuntimeError("Microscope not connected.")
        # PyJEM GetPos returns [x, y, z, tx, ty]
        pos = self._stage.GetPos()
        return StagePosition(
            x_nm=float(pos[0]),
            y_nm=float(pos[1]),
            z_nm=float(pos[2]),
            tx_deg=float(pos[3]),
            ty_deg=float(pos[4]),
        )

    def set_stage_position(self, x_nm: Optional[float] = None, y_nm: Optional[float] = None) -> StagePosition:
        if not self._connected or self._stage is None:
            raise RuntimeError("Microscope not connected.")
        if x_nm is not None and y_nm is not None:
            self._stage.SetPos(float(x_nm), float(y_nm))
        elif x_nm is not None:
            self._stage.SetX(float(x_nm))
        elif y_nm is not None:
            self._stage.SetY(float(y_nm))
        return self.get_stage_position()

    def set_stage_relative(self, dx_nm: float, dy_nm: float) -> StagePosition:
        if not self._connected or self._stage is None:
            raise RuntimeError("Microscope not connected.")
        if dx_nm != 0:
            self._stage.SetXRel(float(dx_nm))
        if dy_nm != 0:
            self._stage.SetYRel(float(dy_nm))
        return self.get_stage_position()

    def get_defocus(self) -> int:
        if not self._connected or self._lens is None:
            raise RuntimeError("Microscope not connected.")
        return int(self._lens.GetOLf())

    def set_defocus(self, value: int) -> None:
        if not self._connected or self._lens is None:
            raise RuntimeError("Microscope not connected.")
        self._lens.SetOLf(int(value))

    def get_beam_blank(self) -> bool:
        if not self._connected or self._def is None:
            raise RuntimeError("Microscope not connected.")
        return bool(self._def.GetBeamBlank())

    def set_beam_blank(self, blank: bool) -> None:
        if not self._connected or self._def is None:
            raise RuntimeError("Microscope not connected.")
        self._def.SetBeamBlank(1 if blank else 0)

    def get_lens_state(self) -> LensState:
        if not self._connected:
            raise RuntimeError("Microscope not connected.")
        olf = self._lens.GetOLf()
        olc = self._lens.GetOLc()
        cl1 = self._lens.GetCL1()
        cl2 = self._lens.GetCL2()
        cl3 = self._lens.GetCL3()
        mag = self._eos.GetMagValue()[0]
        spot = self._eos.GetSpotSize()
        ht = self._ht.GetHtValue()
        return LensState(
            olf=olf, olc=olc, cl1=cl1, cl2=cl2, cl3=cl3, mag=float(mag), spot_size=int(spot), ht_kv=float(ht)
        )

    def get_all_parameters(self) -> Dict[str, Any]:
        pos = self.get_stage_position()
        lens = self.get_lens_state()
        return {
            "stage_x_nm": pos.x_nm,
            "stage_y_nm": pos.y_nm,
            "stage_z_nm": pos.z_nm,
            "tilt_x_deg": pos.tx_deg,
            "tilt_y_deg": pos.ty_deg,
            "magnification": lens.mag,
            "high_tension_kv": lens.ht_kv,
            "spot_size": lens.spot_size,
            "objective_lens_fine": lens.olf,
            "objective_lens_coarse": lens.olc,
            "beam_blank": self.get_beam_blank(),
            "mock_mode": False
        }

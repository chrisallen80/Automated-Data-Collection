"""
Multi-Site Stage Waypoint Management and Batch Execution.
Allows researchers to preselect/record multiple stage positions,
then automatically iterate through each position executing survey,
segmentation, tile planning, and high-resolution acquisition.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from ..hardware.base import StagePosition


class SiteStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class StageSite:
    """Represents a pre-selected specimen stage position / region of interest."""
    site_id: int
    label: str
    x_nm: float
    y_nm: float
    z_nm: float = 0.0
    tx_deg: float = 0.0
    ty_deg: float = 0.0
    status: SiteStatus = SiteStatus.PENDING
    tiles_collected: int = 0
    clusters_found: int = 0
    output_hdf5_path: Optional[str] = None
    error_message: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_stage_position(self) -> StagePosition:
        return StagePosition(
            x_nm=self.x_nm,
            y_nm=self.y_nm,
            z_nm=self.z_nm,
            tx_deg=self.tx_deg,
            ty_deg=self.ty_deg
        )


class MultiSiteManager:
    """
    Manages a queue of preselected stage sites for automated serial data collection.
    """

    def __init__(self):
        self.sites: List[StageSite] = []
        self._next_id: int = 1

    def add_site(
        self,
        x_nm: float,
        y_nm: float,
        z_nm: float = 0.0,
        tx_deg: float = 0.0,
        ty_deg: float = 0.0,
        label: Optional[str] = None
    ) -> StageSite:
        """Create and append a new stage waypoint."""
        site_id = self._next_id
        self._next_id += 1
        name = label if label else f"Site_{site_id:03d}"

        site = StageSite(
            site_id=site_id,
            label=name,
            x_nm=float(x_nm),
            y_nm=float(y_nm),
            z_nm=float(z_nm),
            tx_deg=float(tx_deg),
            ty_deg=float(ty_deg)
        )
        self.sites.append(site)
        return site

    def add_from_position(self, pos: StagePosition, label: Optional[str] = None) -> StageSite:
        """Record current microscope stage position as a new site."""
        return self.add_site(
            x_nm=pos.x_nm,
            y_nm=pos.y_nm,
            z_nm=pos.z_nm,
            tx_deg=pos.tx_deg,
            ty_deg=pos.ty_deg,
            label=label
        )

    def remove_site(self, site_id: int) -> bool:
        """Remove site by ID."""
        initial_len = len(self.sites)
        self.sites = [s for s in self.sites if s.site_id != site_id]
        return len(self.sites) < initial_len

    def clear(self) -> None:
        """Clear all sites."""
        self.sites.clear()
        self._next_id = 1

    def get_site(self, site_id: int) -> Optional[StageSite]:
        for s in self.sites:
            if s.site_id == site_id:
                return s
        return None

    def reset_statuses(self) -> None:
        """Reset all sites to PENDING status for a fresh multi-site run."""
        for s in self.sites:
            s.status = SiteStatus.PENDING
            s.tiles_collected = 0
            s.clusters_found = 0
            s.output_hdf5_path = None
            s.error_message = None

    def save_to_json(self, file_path: str | Path) -> None:
        """Export waypoints to JSON file."""
        data = {
            "version": "2.0",
            "sites": [asdict(s) for s in self.sites]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, file_path: str | Path) -> int:
        """Import waypoints from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_sites = data.get("sites", [])
        self.clear()
        for item in raw_sites:
            site = StageSite(
                site_id=item.get("site_id", self._next_id),
                label=item.get("label", f"Site_{self._next_id}"),
                x_nm=float(item.get("x_nm", 0.0)),
                y_nm=float(item.get("y_nm", 0.0)),
                z_nm=float(item.get("z_nm", 0.0)),
                tx_deg=float(item.get("tx_deg", 0.0)),
                ty_deg=float(item.get("ty_deg", 0.0)),
                status=SiteStatus(item.get("status", SiteStatus.PENDING.value)),
                custom_params=item.get("custom_params", {})
            )
            self.sites.append(site)
            self._next_id = max(self._next_id, site.site_id + 1)

        return len(self.sites)

"""
NeXus / HDF5 Archival Writer for automated electron microscopy acquisitions.
"""

from __future__ import annotations
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import h5py
import numpy as np
from ..imaging.grid_planner import ScanTile


class HDF5Writer:
    """
    Writes standardized HDF5 datasets containing survey micrographs,
    tile coordinates, microscope metadata, and 4D-STEM/virtual detector frames.
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_dataset(
        self,
        survey_image: Optional[np.ndarray],
        tiles: List[ScanTile],
        metadata: Dict[str, Any],
        virtual_reconstructions: Optional[Dict[str, np.ndarray]] = None,
        notes: str = ""
    ) -> Path:
        """
        Create and populate HDF5 container.
        """
        with h5py.File(self.output_path, "w") as f:
            # NeXus attributes
            f.attrs["NX_class"] = "NXroot"
            f.attrs["file_name"] = self.output_path.name
            f.attrs["file_time"] = datetime.datetime.now().isoformat()
            f.attrs["creator"] = "Automated Data Collection (ADC) v2.0"
            if notes:
                f.attrs["notes"] = notes

            # Entry group
            entry = f.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"

            # 1. Instrument / Microscope Metadata group
            instrument = entry.create_group("instrument")
            instrument.attrs["NX_class"] = "NXinstrument"

            microscope_grp = instrument.create_group("microscope")
            for k, v in metadata.items():
                try:
                    if isinstance(v, (int, float, str, bool)):
                        microscope_grp.attrs[k] = v
                    elif isinstance(v, (list, tuple)):
                        microscope_grp.create_dataset(k, data=np.array(v))
                except Exception:
                    pass

            # 2. Survey Image group
            if survey_image is not None:
                survey_grp = entry.create_group("survey")
                survey_grp.attrs["NX_class"] = "NXdata"
                dset = survey_grp.create_dataset(
                    "data", data=survey_image, compression="gzip", compression_opts=4
                )
                dset.attrs["interpretation"] = "image"

            # 3. Scan Tiles group
            if tiles:
                tiles_grp = entry.create_group("tiles")
                t_ids = [t.tile_id for t in tiles]
                c_ids = [t.cluster_id for t in tiles]
                xs = [t.x for t in tiles]
                ys = [t.y for t in tiles]
                ws = [t.width for t in tiles]
                hs = [t.height for t in tiles]
                sx_nm = [t.stage_x_nm for t in tiles]
                sy_nm = [t.stage_y_nm for t in tiles]
                done = [t.completed for t in tiles]

                tiles_grp.create_dataset("tile_id", data=np.array(t_ids, dtype=np.int32))
                tiles_grp.create_dataset("cluster_id", data=np.array(c_ids, dtype=np.int32))
                tiles_grp.create_dataset("pixel_x", data=np.array(xs, dtype=np.int32))
                tiles_grp.create_dataset("pixel_y", data=np.array(ys, dtype=np.int32))
                tiles_grp.create_dataset("width", data=np.array(ws, dtype=np.int32))
                tiles_grp.create_dataset("height", data=np.array(hs, dtype=np.int32))
                tiles_grp.create_dataset("stage_x_nm", data=np.array(sx_nm, dtype=np.float64))
                tiles_grp.create_dataset("stage_y_nm", data=np.array(sy_nm, dtype=np.float64))
                tiles_grp.create_dataset("completed", data=np.array(done, dtype=bool))

            # 4. Virtual Reconstructions (VADF / VBF / COM)
            if virtual_reconstructions:
                recon_grp = entry.create_group("virtual_reconstructions")
                recon_grp.attrs["NX_class"] = "NXdata"
                for name, img_data in virtual_reconstructions.items():
                    dset = recon_grp.create_dataset(
                        name, data=img_data, compression="gzip", compression_opts=4
                    )
                    dset.attrs["interpretation"] = "image"

        return self.output_path

"""
Grid Decomposition, Tile Partitioning, and Bounding Box Tightening.
Transforms detected feature clusters into an optimized queue of high-resolution scan tiles.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from .clustering import Cluster


@dataclass
class ScanTile:
    """Represents a discrete scan tile for high-resolution acquisition."""
    tile_id: int
    cluster_id: int
    x: int                 # Pixel x-coordinate in survey image
    y: int                 # Pixel y-coordinate in survey image
    width: int             # Tile width in pixels
    height: int            # Tile height in pixels
    row: int               # Grid row index
    col: int               # Grid column index
    stage_x_nm: float = 0.0 # Physical stage coordinate
    stage_y_nm: float = 0.0 # Physical stage coordinate
    completed: bool = False


class GridPlanner:
    """
    Partitions regions of interest into overlapping square tiles,
    tightens bounding boxes around actual features, and sorts the tiles
    into optimal snake trajectory order.
    """

    def __init__(
        self,
        tile_size: int = 256,
        overlap_ratio: float = 0.10,
        nm_per_pixel: float = 1.0,
        stage_origin_nm: Tuple[float, float] = (0.0, 0.0)
    ):
        self.tile_size = int(tile_size)
        self.overlap_ratio = float(np.clip(overlap_ratio, 0.0, 0.8))
        self.nm_per_pixel = float(nm_per_pixel)
        self.stage_origin_nm = stage_origin_nm

    def tighten_bbox(
        self,
        cluster: Cluster,
        binary_mask: Optional[np.ndarray] = None,
        padding_px: int = 4
    ) -> Tuple[int, int, int, int]:
        """
        Tighten the bounding box of a cluster so it strictly bounds foreground pixels.
        Avoids wasting beam time scanning empty vacuum.
        """
        x, y, w, h = cluster.bbox

        if binary_mask is not None and w > 0 and h > 0:
            h_img, w_img = binary_mask.shape[:2]
            y0, y1 = max(0, y), min(h_img, y + h)
            x0, x1 = max(0, x), min(w_img, x + w)
            roi = binary_mask[y0:y1, x0:x1]

            ys, xs = np.where(roi > 0)
            if len(xs) > 0 and len(ys) > 0:
                min_x = max(0, x0 + int(np.min(xs)) - padding_px)
                max_x = min(w_img, x0 + int(np.max(xs)) + padding_px)
                min_y = max(0, y0 + int(np.min(ys)) - padding_px)
                max_y = min(h_img, y0 + int(np.max(ys)) + padding_px)
                return (min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y))

        return cluster.bbox

    def plan_cluster_tiles(
        self,
        cluster: Cluster,
        survey_shape: Tuple[int, int],
        binary_mask: Optional[np.ndarray] = None,
        tighten: bool = True
    ) -> List[ScanTile]:
        """
        Generate overlapping scan tiles covering a single cluster.
        """
        if tighten:
            bx, by, bw, bh = self.tighten_bbox(cluster, binary_mask)
        else:
            bx, by, bw, bh = cluster.bbox

        img_h, img_w = survey_shape[:2]

        # Calculate stride with overlap
        stride = int(round(self.tile_size * (1.0 - self.overlap_ratio)))
        stride = max(1, stride)

        # Determine cluster center in x and y
        cx = bx + bw / 2.0
        cy = by + bh / 2.0

        # Number of tiles needed in x (centered around cx)
        if bw <= self.tile_size:
            start_x = int(round(cx - self.tile_size / 2.0))
            x_offsets = [start_x]
        else:
            num_cols = int(np.ceil((bw - self.tile_size) / stride)) + 1
            total_span_x = (num_cols - 1) * stride + self.tile_size
            start_x = int(round(cx - total_span_x / 2.0))
            x_offsets = [start_x + i * stride for i in range(num_cols)]

        # Number of tiles needed in y (centered around cy)
        if bh <= self.tile_size:
            start_y = int(round(cy - self.tile_size / 2.0))
            y_offsets = [start_y]
        else:
            num_rows = int(np.ceil((bh - self.tile_size) / stride)) + 1
            total_span_y = (num_rows - 1) * stride + self.tile_size
            start_y = int(round(cy - total_span_y / 2.0))
            y_offsets = [start_y + j * stride for j in range(num_rows)]

        tiles: List[ScanTile] = []
        tile_count = 0

        # Snake ordering: alternate column traversal direction on each row
        for row_idx, ty in enumerate(y_offsets):
            cols = list(enumerate(x_offsets))
            if row_idx % 2 == 1:
                cols.reverse()

            for col_idx, tx in cols:
                # Clamp within survey image dimensions
                clamped_x = int(np.clip(tx, 0, max(0, img_w - self.tile_size)))
                clamped_y = int(np.clip(ty, 0, max(0, img_h - self.tile_size)))

                # Map pixel coordinates to physical stage nanometers
                sx_nm = self.stage_origin_nm[0] + (clamped_x + self.tile_size / 2.0) * self.nm_per_pixel
                sy_nm = self.stage_origin_nm[1] + (clamped_y + self.tile_size / 2.0) * self.nm_per_pixel

                tiles.append(ScanTile(
                    tile_id=tile_count,
                    cluster_id=cluster.id,
                    x=clamped_x,
                    y=clamped_y,
                    width=self.tile_size,
                    height=self.tile_size,
                    row=row_idx,
                    col=col_idx,
                    stage_x_nm=sx_nm,
                    stage_y_nm=sy_nm
                ))
                tile_count += 1

        return tiles

    def plan_all(
        self,
        clusters: List[Cluster],
        survey_shape: Tuple[int, int],
        binary_mask: Optional[np.ndarray] = None,
        tighten: bool = True
    ) -> List[ScanTile]:
        """
        Generate and concatenate tiles for all active/selected clusters.
        """
        all_tiles: List[ScanTile] = []
        global_id = 0

        for cluster in clusters:
            if not cluster.selected:
                continue
            c_tiles = self.plan_cluster_tiles(cluster, survey_shape, binary_mask, tighten=tighten)
            for t in c_tiles:
                t.tile_id = global_id
                global_id += 1
                all_tiles.append(t)

        return all_tiles

"""
Scan pattern generators including raster, snake, spiral, and arbitrary custom trajectories.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np


class PatternGenerator:
    """
    Generates probe scanning coordinates for 2D scan coils or stage positioning.
    All patterns return ordered lists of 2D coordinates (x, y).
    """

    @staticmethod
    def raster(width: int, height: int, step: int = 1) -> List[Tuple[int, int]]:
        """Standard unidirectional raster scan (flyback at end of each line)."""
        coords = []
        for y in range(0, height, step):
            for x in range(0, width, step):
                coords.append((x, y))
        return coords

    @staticmethod
    def snake(width: int, height: int, step: int = 1) -> List[Tuple[int, int]]:
        """Bidirectional snake raster (alternates direction each row to minimize coil flyback)."""
        coords = []
        for row_idx, y in enumerate(range(0, height, step)):
            xs = list(range(0, width, step))
            if row_idx % 2 == 1:
                xs.reverse()
            for x in xs:
                coords.append((x, y))
        return coords

    @staticmethod
    def spiral(size: int, step: float = 1.0, turns: int = 10) -> List[Tuple[float, float]]:
        """
        Archimedean spiral from center outwards.
        Reduces localized charging and contamination on beam-sensitive samples.
        """
        cx, cy = size / 2.0, size / 2.0
        max_r = size / 2.0
        coords = []

        total_pts = int(size * turns)
        theta_max = turns * 2.0 * np.pi
        thetas = np.linspace(0, theta_max, total_pts)

        for th in thetas:
            r = (th / theta_max) * max_r
            x = cx + r * np.cos(th)
            y = cy + r * np.sin(th)
            if 0 <= x < size and 0 <= y < size:
                coords.append((round(x, 2), round(y, 2)))

        return coords

    @staticmethod
    def sparse_random(width: int, height: int, coverage_fraction: float = 0.1, seed: int = 42) -> List[Tuple[int, int]]:
        """
        Sparse random sampling (compressive sensing STEM).
        Acquires a random fraction of total pixels to reduce electron dose.
        """
        num_pixels = int(width * height * np.clip(coverage_fraction, 0.01, 1.0))
        rng = np.random.default_rng(seed)
        all_indices = rng.choice(width * height, size=num_pixels, replace=False)

        coords = []
        for idx in all_indices:
            x = int(idx % width)
            y = int(idx // width)
            coords.append((x, y))
        return coords

    @staticmethod
    def arbitrary(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Pass-through validator for user-defined arbitrary trajectory coordinates."""
        return [(float(pt[0]), float(pt[1])) for pt in points]

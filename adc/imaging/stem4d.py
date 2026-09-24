"""
4D-STEM Diffraction Processing and Virtual Detector Imaging.
Enables using the Merlin detector for survey imaging, virtual aperture synthesis
(VADF, VBF, VABF), differential phase contrast (Center of Mass), and Bragg crystallinity mapping.
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import cv2


class STEM4DProcessor:
    """
    High-performance tools for processing 4D-STEM datacubes (Ny, Nx, Ky, Kx)
    or streaming individual 2D diffraction frames (Ky, Kx).
    """

    @staticmethod
    def create_circular_mask(
        shape: Tuple[int, int],
        inner_radius: float = 0.0,
        outer_radius: float = 50.0,
        center: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """Create a boolean annular or circular aperture mask."""
        h, w = shape
        if center is None:
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        else:
            cx, cy = center

        y, x = np.ogrid[:h, :w]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        mask = (dist_sq >= inner_radius ** 2) & (dist_sq <= outer_radius ** 2)
        return mask

    @classmethod
    def integrate_virtual_aperture(
        cls,
        diffraction_frame: np.ndarray,
        inner_radius: float,
        outer_radius: float,
        center: Optional[Tuple[float, float]] = None
    ) -> float:
        """
        Integrate intensity of a single 2D diffraction pattern within an annular aperture.
        """
        mask = cls.create_circular_mask(diffraction_frame.shape, inner_radius, outer_radius, center)
        return float(np.sum(diffraction_frame[mask]))

    @classmethod
    def reconstruct_vadf_cube(
        cls,
        datacube: np.ndarray,
        inner_radius: float,
        outer_radius: float,
        center: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Reconstruct a 2D Virtual Annular Dark Field (VADF) image from a 4D-STEM datacube (Ny, Nx, Ky, Kx).
        """
        ny, nx, ky, kx = datacube.shape
        mask = cls.create_circular_mask((ky, kx), inner_radius, outer_radius, center)
        # Vectorized sum over k-space dimensions
        vadf_img = np.sum(datacube[:, :, mask], axis=-1)
        return vadf_img.astype(np.float32)

    @classmethod
    def reconstruct_vbf_cube(
        cls,
        datacube: np.ndarray,
        disk_radius: float,
        center: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Reconstruct a 2D Virtual Bright Field (VBF) image from a 4D-STEM datacube.
        """
        return cls.reconstruct_vadf_cube(datacube, 0.0, disk_radius, center)

    @staticmethod
    def compute_center_of_mass(
        diffraction_frame: np.ndarray,
        center: Optional[Tuple[float, float]] = None,
        threshold: float = 0.0
    ) -> Tuple[float, float]:
        """
        Compute Center of Mass (COM / DPC momentum transfer) of a diffraction pattern.
        Returns (com_x, com_y) relative to detector center.
        """
        h, w = diffraction_frame.shape
        if center is None:
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        else:
            cx, cy = center

        pattern = diffraction_frame.astype(np.float32)
        if threshold > 0:
            pattern = np.where(pattern >= threshold, pattern, 0.0)

        total = np.sum(pattern)
        if total <= 1e-9:
            return 0.0, 0.0

        y_indices, x_indices = np.indices((h, w))
        com_x = float(np.sum(x_indices * pattern) / total) - cx
        com_y = float(np.sum(y_indices * pattern) / total) - cy
        return com_x, com_y

    @classmethod
    def compute_crystallinity_index(
        cls,
        diffraction_frame: np.ndarray,
        central_disk_radius: Optional[float] = None,
        outer_annulus_radius: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze whether a diffraction frame exhibits sharp crystalline Bragg reflections
        or merely diffuse amorphous / vacuum background.
        Used by autonomous decision routines to skip empty carbon film or target crystal grains.
        """
        h, w = diffraction_frame.shape
        if center is None:
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        else:
            cx, cy = center

        if central_disk_radius is None:
            central_disk_radius = min(h, w) * 0.15
        if outer_annulus_radius is None:
            outer_annulus_radius = min(h, w) * 0.45

        # Mask outside the primary transmitted beam disc
        annulus_mask = cls.create_circular_mask((h, w), central_disk_radius, outer_annulus_radius, (cx, cy))
        central_mask = cls.create_circular_mask((h, w), 0.0, central_disk_radius, (cx, cy))

        central_intensity = float(np.sum(diffraction_frame[central_mask]))
        annulus_region = diffraction_frame * annulus_mask

        # Detect discrete local maxima (Bragg reflections) in the annulus
        annulus_max = float(np.max(annulus_region)) if np.any(annulus_mask) else 0.0
        annulus_mean = float(np.mean(diffraction_frame[annulus_mask])) if np.any(annulus_mask) else 0.0

        # Peak-to-background ratio in diffraction annulus
        pbr = (annulus_max / (annulus_mean + 1e-6)) if annulus_mean > 0 else 0.0

        # Bragg peak detection using SimpleBlobDetector or local thresholding
        threshold_val = annulus_mean + 3.0 * float(np.std(diffraction_frame[annulus_mask]))
        peaks_mask = (annulus_region > threshold_val)
        num_peak_pixels = int(np.sum(peaks_mask))

        is_crystalline = (pbr > 4.5 and num_peak_pixels >= 10)

        return {
            "is_crystalline": bool(is_crystalline),
            "peak_to_background_ratio": float(pbr),
            "annulus_mean": annulus_mean,
            "annulus_max": annulus_max,
            "central_intensity": central_intensity,
            "peak_pixel_count": num_peak_pixels
        }

"""
Phase Cross-Correlation Drift Tracking and Compensation.
Tracks sub-pixel specimen drift against a reference patch and updates scan targets.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.filters import window

from .base import BaseCorrectionRoutine, CorrectionLocationMode, CorrectionContext, CorrectionResult


@dataclass
class DriftMeasurement:
    """Record of a single drift correction step."""
    step_index: int
    dx_px: float
    dy_px: float
    dx_total_nm: float
    dy_total_nm: float
    error: float


class DriftTracker:
    """
    Sub-pixel specimen drift tracker using Fourier Phase Cross-Correlation.
    Monitors a designated reference region of interest (ROI) over time.
    """

    def __init__(
        self,
        ref_bbox: Optional[Tuple[int, int, int, int]] = None, # (x, y, w, h)
        upsample_factor: int = 10,
        nm_per_pixel: float = 1.0
    ):
        self.ref_bbox = ref_bbox
        self.upsample_factor = upsample_factor
        self.nm_per_pixel = nm_per_pixel

        self._reference_patch: Optional[np.ndarray] = None
        self._total_drift_px = np.array([0.0, 0.0], dtype=np.float64) # [dy, dx]
        self.history: List[DriftMeasurement] = []

    def reset(self) -> None:
        """Reset reference patch and accumulated drift history."""
        self._reference_patch = None
        self._total_drift_px = np.array([0.0, 0.0], dtype=np.float64)
        self.history.clear()

    def set_reference(self, survey_image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> None:
        """
        Extract and store the reference patch from the survey image.
        """
        if bbox is not None:
            self.ref_bbox = bbox
        elif self.ref_bbox is None:
            # Default to central 128x128 patch
            h, w = survey_image.shape[:2]
            pw, ph = min(128, w // 2), min(128, h // 2)
            self.ref_bbox = ((w - pw) // 2, (h - ph) // 2, pw, ph)

        rx, ry, rw, rh = self.ref_bbox
        patch = survey_image[ry:ry + rh, rx:rx + rw].astype(np.float32)

        # Apply 2D Hann window to suppress edge artifacts in FFT
        hann = window("hann", patch.shape)
        self._reference_patch = patch * hann
        self._total_drift_px = np.array([0.0, 0.0], dtype=np.float64)
        self.history.clear()

    def measure_drift(self, current_image: np.ndarray) -> Tuple[float, float]:
        """
        Compare current frame against reference patch to measure drift displacement.
        Returns (dx_nm, dy_nm).
        """
        if self._reference_patch is None or self.ref_bbox is None:
            return 0.0, 0.0

        rx, ry, rw, rh = self.ref_bbox
        # Crop region around expected reference location (with margin)
        margin = 32
        h, w = current_image.shape[:2]
        cy0 = max(0, ry - margin)
        cy1 = min(h, ry + rh + margin)
        cx0 = max(0, rx - margin)
        cx1 = min(w, rx + rw + margin)

        # We crop the nominal reference location from the new image
        current_patch = current_image[ry:ry + rh, rx:rx + rw].astype(np.float32)
        if current_patch.shape != self._reference_patch.shape:
            return 0.0, 0.0

        hann = window("hann", current_patch.shape)
        windowed_current = current_patch * hann

        # Phase cross-correlation with subpixel registration
        # shifts vector is the translation required to bring current back into registration with reference
        shifts, error, phasediff = phase_cross_correlation(
            self._reference_patch,
            windowed_current,
            upsample_factor=self.upsample_factor
        )

        # Specimen drift is opposite of registration shift vector
        dy_px, dx_px = -float(shifts[0]), -float(shifts[1])

        self._total_drift_px[0] += dy_px
        self._total_drift_px[1] += dx_px

        total_dx_nm = float(self._total_drift_px[1] * self.nm_per_pixel)
        total_dy_nm = float(self._total_drift_px[0] * self.nm_per_pixel)

        measurement = DriftMeasurement(
            step_index=len(self.history) + 1,
            dx_px=dx_px,
            dy_px=dy_px,
            dx_total_nm=total_dx_nm,
            dy_total_nm=total_dy_nm,
            error=float(error)
        )
        self.history.append(measurement)

        return total_dx_nm, total_dy_nm

    def get_drift_corrected_coords(self, x_nm: float, y_nm: float) -> Tuple[float, float]:
        """
        Apply accumulated drift offset to target stage coordinates.
        """
        dx_total_nm = self._total_drift_px[1] * self.nm_per_pixel
        dy_total_nm = self._total_drift_px[0] * self.nm_per_pixel
        return (x_nm + dx_total_nm, y_nm + dy_total_nm)


class DriftCorrectionRoutine(BaseCorrectionRoutine):
    """
    Pluggable in-flight drift correction routine.
    Evaluates drift frequency (every N scans) and reference location strategy
    (previous scan, next scan, or fixed cluster region).
    """

    def __init__(
        self,
        tracker: Optional[DriftTracker] = None,
        enabled: bool = True,
        interval_scans: int = 5,
        location_mode: CorrectionLocationMode = CorrectionLocationMode.FIXED_REGION,
        fixed_cluster_id: Optional[int] = None,
        arbitrary_position_nm: Optional[Tuple[float, float]] = None,
        arbitrary_position_px: Optional[Tuple[int, int]] = None
    ):
        super().__init__(
            name="drift",
            enabled=enabled,
            interval_scans=interval_scans,
            location_mode=location_mode,
            fixed_cluster_id=fixed_cluster_id,
            arbitrary_position_nm=arbitrary_position_nm,
            arbitrary_position_px=arbitrary_position_px
        )
        self.tracker = tracker or DriftTracker()

    def execute(self, context: CorrectionContext) -> CorrectionResult:
        if not self.enabled:
            return CorrectionResult(routine_name=self.name, executed=False)

        # Ensure reference is initialized if available
        if self.tracker._reference_patch is None and context.survey_image is not None:
            if self.location_mode == CorrectionLocationMode.FIXED_REGION and context.fixed_cluster is not None:
                self.tracker.set_reference(context.survey_image, bbox=context.fixed_cluster.bbox)
            elif self.location_mode == CorrectionLocationMode.ARBITRARY_POSITION:
                target_px = self.arbitrary_position_px or context.arbitrary_position_px
                if target_px is not None:
                    px, py = int(target_px[0]), int(target_px[1])
                    h, w = context.survey_image.shape[:2]
                    bw, bh = 64, 64
                    bx = max(0, min(w - bw, px - bw // 2))
                    by = max(0, min(h - bh, py - bh // 2))
                    self.tracker.set_reference(context.survey_image, bbox=(bx, by, bw, bh))
                else:
                    self.tracker.set_reference(context.survey_image)
            else:
                self.tracker.set_reference(context.survey_image)

        # Acquire probe image based on location mode
        probe_image = None
        orig_pos = context.microscope.get_stage_position()

        if self.location_mode == CorrectionLocationMode.FIXED_REGION and context.fixed_cluster is not None:
            # Drive temporarily to fixed reference cluster
            fc = context.fixed_cluster
            # Drive to cluster center
            cluster_stage_x = orig_pos.x_nm + (fc.centroid[0] - context.survey_image.shape[1] / 2) * context.nm_per_pixel
            cluster_stage_y = orig_pos.y_nm + (fc.centroid[1] - context.survey_image.shape[0] / 2) * context.nm_per_pixel
            context.microscope.set_stage_position(cluster_stage_x, cluster_stage_y)
            probe_image = context.scanner.acquire_survey_image(size=(128, 128))
            # Return stage to tile
            context.microscope.set_stage_position(orig_pos.x_nm, orig_pos.y_nm)

        elif self.location_mode == CorrectionLocationMode.ARBITRARY_POSITION:
            # Drive temporarily to arbitrary user-specified coordinate
            target_nm = self.arbitrary_position_nm or context.arbitrary_position_nm
            target_px = self.arbitrary_position_px or context.arbitrary_position_px

            if target_nm is not None:
                arb_x, arb_y = target_nm
            elif target_px is not None and context.survey_image is not None:
                arb_x = orig_pos.x_nm + (target_px[0] - context.survey_image.shape[1] / 2) * context.nm_per_pixel
                arb_y = orig_pos.y_nm + (target_px[1] - context.survey_image.shape[0] / 2) * context.nm_per_pixel
            else:
                arb_x, arb_y = orig_pos.x_nm, orig_pos.y_nm

            context.microscope.set_stage_position(arb_x, arb_y)
            probe_image = context.scanner.acquire_survey_image(size=(128, 128))
            # Return stage to tile
            context.microscope.set_stage_position(orig_pos.x_nm, orig_pos.y_nm)

        elif self.location_mode == CorrectionLocationMode.PREVIOUS_SCAN and context.previous_tile is not None:
            # Check drift against previous tile
            probe_image = context.scanner.acquire_survey_image(size=(128, 128))

        else: # NEXT_SCAN or default
            # Acquire quick probe scan at current upcoming tile location
            probe_image = context.scanner.acquire_survey_image(size=(128, 128))

        if probe_image is None or self.tracker._reference_patch is None:
            return CorrectionResult(
                routine_name=self.name,
                executed=False,
                message="Drift reference or probe image unavailable."
            )

        # Measure drift
        dx_nm, dy_nm = self.tracker.measure_drift(probe_image)

        return CorrectionResult(
            routine_name=self.name,
            executed=True,
            dx_nm=dx_nm,
            dy_nm=dy_nm,
            message=f"Drift measured: dx={dx_nm:.2f} nm, dy={dy_nm:.2f} nm (Mode: {self.location_mode.value}, every {self.interval_scans} scans)"
        )


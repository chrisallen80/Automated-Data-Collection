"""
Automated Defocus Sweep and Sharpness Optimization (Autofocus).
"""

from __future__ import annotations
from typing import List, Tuple, Callable, Optional
import numpy as np
import cv2
from ..hardware.base import BaseMicroscope, BaseScanner
from .base import BaseCorrectionRoutine, CorrectionLocationMode, CorrectionContext, CorrectionResult


class AutofocusController:
    """
    Automates objective lens fine (OLf) focus sweeps and sharpness peak detection.
    Evaluates image focus quality via Laplacian variance and gradient Tenengrad metrics.
    """

    def __init__(
        self,
        microscope: BaseMicroscope,
        scanner: BaseScanner,
        sweep_range: int = 50,
        sweep_steps: int = 7
    ):
        self.microscope = microscope
        self.scanner = scanner
        self.sweep_range = sweep_range
        self.sweep_steps = max(3, sweep_steps if sweep_steps % 2 == 1 else sweep_steps + 1)

    @staticmethod
    def calculate_sharpness_score(image: np.ndarray) -> float:
        """
        Compute image sharpness score using modified Tenengrad gradient variance.
        Higher score corresponds to sharper contrast / higher spatial frequencies.
        """
        if image is None or image.size == 0:
            return 0.0

        # Laplacian variance
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        score_lap = float(laplacian.var())

        # Sobel gradient magnitude
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        score_sob = float(np.mean(gx ** 2 + gy ** 2))

        return score_lap + score_sob * 0.1

    def run_autofocus(self, target_size: Tuple[int, int] = (256, 256)) -> Tuple[int, float]:
        """
        Execute focus sweep around current OLf, fit parabola to scores,
        and set the microscope to the optimal focus value.
        Returns (optimal_olf, best_score).
        """
        current_olf = self.microscope.get_defocus()
        step_size = (self.sweep_range * 2) // (self.sweep_steps - 1)

        olf_values = [
            current_olf - self.sweep_range + i * step_size
            for i in range(self.sweep_steps)
        ]

        scores: List[float] = []

        for val in olf_values:
            self.microscope.set_defocus(val)
            # Short dwell frame acquisition
            frame = self.scanner.acquire_survey_image(size=target_size, dwell_time_s=5e-6)
            score = self.calculate_sharpness_score(frame)
            scores.append(score)

        # Fit quadratic curve: y = a*x^2 + b*x + c
        poly_coeffs = np.polyfit(olf_values, scores, deg=2)
        a, b, c = poly_coeffs

        if a < 0:
            # Vertex of concave parabola gives maximum sharpness: x_opt = -b / (2a)
            optimal_olf = int(round(-b / (2.0 * a)))
            # Clamp within sweep range
            optimal_olf = max(min(olf_values), min(max(olf_values), optimal_olf))
        else:
            # If fit is not concave, take discrete argmax
            best_idx = int(np.argmax(scores))
            optimal_olf = olf_values[best_idx]

        # Apply optimal focus to microscope
        self.microscope.set_defocus(optimal_olf)
        best_score = float(max(scores))

        return optimal_olf, best_score


class AutofocusCorrectionRoutine(BaseCorrectionRoutine):
    """
    Pluggable in-flight autofocus correction routine.
    Evaluates focus frequency (every M scans) and reference location strategy
    (previous scan, next scan, or fixed cluster region).
    """

    def __init__(
        self,
        controller: Optional[AutofocusController] = None,
        enabled: bool = False,
        interval_scans: int = 10,
        location_mode: CorrectionLocationMode = CorrectionLocationMode.NEXT_SCAN,
        fixed_cluster_id: Optional[int] = None,
        arbitrary_position_nm: Optional[Tuple[float, float]] = None,
        arbitrary_position_px: Optional[Tuple[int, int]] = None,
        sweep_range: int = 50,
        sweep_steps: int = 7
    ):
        super().__init__(
            name="focus",
            enabled=enabled,
            interval_scans=interval_scans,
            location_mode=location_mode,
            fixed_cluster_id=fixed_cluster_id,
            arbitrary_position_nm=arbitrary_position_nm,
            arbitrary_position_px=arbitrary_position_px
        )
        self.controller = controller
        self.sweep_range = sweep_range
        self.sweep_steps = sweep_steps

    def execute(self, context: CorrectionContext) -> CorrectionResult:
        if not self.enabled:
            return CorrectionResult(routine_name=self.name, executed=False)

        ctrl = self.controller or AutofocusController(
            context.microscope,
            context.scanner,
            sweep_range=self.sweep_range,
            sweep_steps=self.sweep_steps
        )

        initial_olf = context.microscope.get_defocus()
        orig_pos = context.microscope.get_stage_position()

        # Handle location mode
        if self.location_mode == CorrectionLocationMode.FIXED_REGION and context.fixed_cluster is not None:
            fc = context.fixed_cluster
            cluster_stage_x = orig_pos.x_nm + (fc.centroid[0] - context.survey_image.shape[1] / 2) * context.nm_per_pixel
            cluster_stage_y = orig_pos.y_nm + (fc.centroid[1] - context.survey_image.shape[0] / 2) * context.nm_per_pixel
            context.microscope.set_stage_position(cluster_stage_x, cluster_stage_y)
            optimal_olf, score = ctrl.run_autofocus()
            context.microscope.set_stage_position(orig_pos.x_nm, orig_pos.y_nm)

        elif self.location_mode == CorrectionLocationMode.ARBITRARY_POSITION:
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
            optimal_olf, score = ctrl.run_autofocus()
            context.microscope.set_stage_position(orig_pos.x_nm, orig_pos.y_nm)

        elif self.location_mode == CorrectionLocationMode.PREVIOUS_SCAN and context.previous_tile is not None:
            optimal_olf, score = ctrl.run_autofocus()

        else: # NEXT_SCAN
            optimal_olf, score = ctrl.run_autofocus()

        delta = optimal_olf - initial_olf
        return CorrectionResult(
            routine_name=self.name,
            executed=True,
            defocus_delta=float(delta),
            message=f"Autofocus optimized: OLf={optimal_olf} (Δ={delta:+d}, Score={score:.1f}, Mode={self.location_mode.value}, every {self.interval_scans} scans)"
        )


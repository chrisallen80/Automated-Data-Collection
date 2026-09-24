"""
Decision routine filtering candidate scan targets.
"""

from __future__ import annotations
from typing import List, Dict, Any
from ..base import SurveyData, ScanTarget
from ..loader import adc_decision
from ...imaging.clustering import FeatureClusterer


@adc_decision
def decide_high_contrast_targets(survey: SurveyData, config: Dict[str, Any]) -> List[ScanTarget]:
    """
    Selects scan targets from survey image based on feature contrast and size.
    Sorts targets by descending priority (e.g. largest, highest-contrast features first).
    """
    target_tile_size = int(config.get("tile_size", 256))
    min_intensity = float(config.get("min_intensity", 100.0))

    # Binary mask of high-contrast regions
    mask = (survey.image > min_intensity).astype("uint8") * 255
    clusterer = FeatureClusterer(min_area=50, max_area=100000)
    clusters = clusterer.segment_contours(mask, original_img=survey.image)

    targets: List[ScanTarget] = []
    for c in clusters:
        # Map pixel to stage
        sx = survey.stage_x_nm + (c.center[0] - survey.image.shape[1] / 2.0) * survey.nm_per_pixel
        sy = survey.stage_y_nm + (c.center[1] - survey.image.shape[0] / 2.0) * survey.nm_per_pixel

        targets.append(ScanTarget(
            target_id=c.id,
            x_px=int(c.center[0] - target_tile_size // 2),
            y_px=int(c.center[1] - target_tile_size // 2),
            width_px=target_tile_size,
            height_px=target_tile_size,
            stage_x_nm=sx,
            stage_y_nm=sy,
            dwell_time_s=float(config.get("dwell_s", 1e-3)),
            priority=int(c.area_px) # Higher area = higher priority
        ))

    # Sort descending by priority
    targets.sort(key=lambda t: t.priority, reverse=True)
    return targets

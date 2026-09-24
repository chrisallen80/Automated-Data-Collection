"""
Adaptive live feedback routine: steers dwell time or skips empty tiles dynamically.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
from ..base import ScanTarget, LiveDecision, DecisionAction
from ..loader import adc_live_feedback
from ...imaging.stem4d import STEM4DProcessor


@adc_live_feedback
def feedback_adaptive_dwell(
    diffraction_frame: np.ndarray,
    target: ScanTarget,
    config: Dict[str, Any]
) -> LiveDecision:
    """
    Live decision hook executed during 4D-STEM acquisition.
    Examines incoming diffraction pattern:
    - If strong crystalline diffraction is found: increases dwell time for high signal-to-noise.
    - If empty background: skips remainder or reduces dwell time.
    """
    result = STEM4DProcessor.compute_crystallinity_index(diffraction_frame)

    if result["is_crystalline"]:
        # Lengthen dwell time to get high signal-to-noise on high-index Bragg reflections
        boosted_dwell = target.dwell_time_s * 2.0
        return LiveDecision(
            action=DecisionAction.ADJUST_DWELL,
            new_dwell_s=boosted_dwell,
            reason=f"Crystalline reflections detected (PBR={result['peak_to_background_ratio']:.1f}), boosting dwell.",
            metadata=result
        )
    elif result["central_intensity"] < 100.0:
        # Beam is blocked or empty vacuum
        return LiveDecision(
            action=DecisionAction.SKIP,
            reason="Low transmitted intensity (vacuum or blocked beam).",
            metadata=result
        )

    return LiveDecision(
        action=DecisionAction.ACQUIRE,
        reason="Normal amorphous/specimen diffraction.",
        metadata=result
    )

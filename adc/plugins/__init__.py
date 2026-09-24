"""
Extensible plugin and custom routine system.
"""

from .base import (
    DecisionAction,
    SurveyData,
    ScanTarget,
    LiveDecision,
    ClusteringFunc,
    DecisionFunc,
    LiveFeedbackFunc,
)
from .loader import (
    PluginRegistry,
    GLOBAL_REGISTRY,
    adc_clustering,
    adc_decision,
    adc_live_feedback,
    adc_correction,
)
from .builtins import (
    cluster_dbscan_routine,
    decide_high_contrast_targets,
    feedback_adaptive_dwell,
)

# Register built-ins in the global registry
GLOBAL_REGISTRY.register_clustering("dbscan", cluster_dbscan_routine)
GLOBAL_REGISTRY.register_decision("high_contrast", decide_high_contrast_targets)
GLOBAL_REGISTRY.register_live_feedback("adaptive_dwell", feedback_adaptive_dwell)

__all__ = [
    "DecisionAction",
    "SurveyData",
    "ScanTarget",
    "LiveDecision",
    "ClusteringFunc",
    "DecisionFunc",
    "LiveFeedbackFunc",
    "adc_clustering",
    "adc_decision",
    "adc_live_feedback",
    "PluginRegistry",
    "GLOBAL_REGISTRY",
    "cluster_dbscan_routine",
    "decide_high_contrast_targets",
    "feedback_adaptive_dwell",
]

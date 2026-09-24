"""
Built-in DBSCAN clustering routine example.
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from ..loader import adc_clustering
from ...imaging.clustering import FeatureClusterer, Cluster


@adc_clustering
def cluster_dbscan_routine(image: np.ndarray, config: Dict[str, Any]) -> List[Cluster]:
    """
    Segmentation routine using DBSCAN on thresholded foreground pixels.
    """
    eps = float(config.get("eps", 3.0))
    min_samples = int(config.get("min_samples", 5))
    min_area = int(config.get("min_area", 20))
    max_area = int(config.get("max_area", 500000))

    clusterer = FeatureClusterer(min_area=min_area, max_area=max_area, eps=eps, min_samples=min_samples)
    return clusterer.segment_dbscan(image)

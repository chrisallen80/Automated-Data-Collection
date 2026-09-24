"""
Feature segmentation and clustering algorithms for STEM images.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import cv2
from sklearn.cluster import DBSCAN


@dataclass
class Cluster:
    """Represents a segmented region of interest (e.g. nanoparticle, domain, feature)."""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]     # (cx, cy)
    area_px: int
    mean_intensity: float = 0.0
    contour: Optional[np.ndarray] = None
    selected: bool = True           # User can toggle individual clusters for scanning

    @property
    def cluster_id(self) -> int:
        return self.id

    @property
    def centroid(self) -> Tuple[float, float]:
        return self.center


class FeatureClusterer:
    """
    Segments regions of interest using either:
    1. Connected components / Contours (fast, deterministic)
    2. DBSCAN on foreground pixel coordinates (robust to fragmented clusters)
    """

    def __init__(
        self,
        min_area: int = 20,
        max_area: int = 500000,
        eps: float = 3.0,
        min_samples: int = 5
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.eps = eps
        self.min_samples = min_samples

    def segment_contours(self, binary_mask: np.ndarray, original_img: Optional[np.ndarray] = None) -> List[Cluster]:
        """
        Segment connected components from a binary mask using OpenCV findContours.
        """
        if binary_mask is None or binary_mask.size == 0:
            return []

        # Find external contours
        contours, hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        clusters: List[Cluster] = []
        c_id = 0
        h_img, w_img = binary_mask.shape[:2]

        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if not (self.min_area <= area <= self.max_area):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            
            # Reject clusters that touch the absolute edges of the survey image
            border_margin = 2
            if x <= border_margin or y <= border_margin or (x + w) >= (w_img - border_margin) or (y + h) >= (h_img - border_margin):
                continue

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx, cy = float(x + w / 2), float(y + h / 2)

            # Compute mean intensity from original grayscale image if provided
            mean_val = 0.0
            if original_img is not None:
                roi = original_img[y:y + h, x:x + w]
                mean_val = float(np.mean(roi))

            clusters.append(Cluster(
                id=c_id,
                bbox=(x, y, w, h),
                center=(cx, cy),
                area_px=area,
                mean_intensity=mean_val,
                contour=cnt
            ))
            c_id += 1

        return clusters

    def segment_dbscan(self, binary_mask: np.ndarray, original_img: Optional[np.ndarray] = None) -> List[Cluster]:
        """
        Segment foreground pixels using Scikit-Learn DBSCAN clustering.
        Ideal for sparse, fragmented, or noisy clusters.
        """
        if binary_mask is None or binary_mask.size == 0:
            return []

        # Find all foreground pixel coordinates (y, x)
        y_indices, x_indices = np.where(binary_mask > 0)
        if len(x_indices) == 0:
            return []

        # Subsample if pixel count is too large (> 15000) for fast DBSCAN
        max_pts = 15000
        if len(x_indices) > max_pts:
            step = len(x_indices) // max_pts
            coords = np.column_stack((x_indices[::step], y_indices[::step]))
        else:
            coords = np.column_stack((x_indices, y_indices))

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = db.fit_predict(coords)

        clusters: List[Cluster] = []
        unique_labels = set(labels) - {-1} # Ignore noise label -1
        c_id = 0

        for label in sorted(unique_labels):
            pts = coords[labels == label]
            if len(pts) < self.min_samples:
                continue

            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            w = int(x_max - x_min + 1)
            h = int(y_max - y_min + 1)
            area = len(pts)

            # Reject clusters that touch the absolute edges of the survey image
            border_margin = 2
            if x_min <= border_margin or y_min <= border_margin or (x_max) >= (w_img - border_margin) or (y_max) >= (h_img - border_margin):
                continue

            if not (self.min_area <= area <= self.max_area):
                continue

            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

            mean_val = 0.0
            if original_img is not None:
                roi = original_img[y_min:y_min + h, x_min:x_min + w]
                mean_val = float(np.mean(roi)) if roi.size > 0 else 0.0

            clusters.append(Cluster(
                id=c_id,
                bbox=(int(x_min), int(y_min), w, h),
                center=(cx, cy),
                area_px=area,
                mean_intensity=mean_val
            ))
            c_id += 1

        return clusters

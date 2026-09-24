"""
Modular, configurable image processing and filtering pipeline for STEM micrographs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2


@dataclass
class FilterStep:
    """Represents a single step in the image filter chain."""
    name: str              # e.g., 'gaussian_blur', 'sharpen', 'threshold'
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


class ImagePipeline:
    """
    Configurable image pre-processing pipeline.
    Transforms raw grayscale STEM survey images into binary or segmented feature masks.
    Supports clean JSON/dictionary configuration.
    """

    DEFAULT_STEPS = [
        FilterStep(name="gaussian_blur", enabled=True, params={"ksize": 5, "sigma": 1.5}),
        FilterStep(name="sharpen", enabled=False, params={"strength": 1.0}),
        FilterStep(name="median_blur", enabled=False, params={"ksize": 5}),
        FilterStep(name="threshold", enabled=True, params={"method": "dual", "lower": 120, "upper": 255}),
        FilterStep(name="morph_open", enabled=True, params={"ksize": 3, "shape": "rect", "iterations": 1}),
        FilterStep(name="morph_close", enabled=True, params={"ksize": 3, "shape": "rect", "iterations": 1}),
    ]

    def __init__(self, steps: Optional[List[FilterStep]] = None):
        self.steps: List[FilterStep] = steps if steps is not None else [
            FilterStep(s.name, s.enabled, dict(s.params)) for s in self.DEFAULT_STEPS
        ]
        self.custom_filters: Dict[str, Any] = {}

    def register_filter(self, name: str, func: Any) -> None:
        """Register a custom image filter, thresholding, or morphological routine."""
        self.custom_filters[name.lower()] = func

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> ImagePipeline:
        """Construct pipeline from serialized configuration dictionary."""
        steps = []
        raw_steps = config.get("steps", [])
        for item in raw_steps:
            steps.append(FilterStep(
                name=item.get("name", ""),
                enabled=item.get("enabled", True),
                params=item.get("params", {})
            ))
        return cls(steps)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline configuration to dictionary."""
        return {
            "steps": [
                {"name": s.name, "enabled": s.enabled, "params": s.params}
                for s in self.steps
            ]
        }

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Execute filter chain sequentially on grayscale input image.
        Returns processed image (uint8).
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")

        # Ensure uint8 grayscale
        if image.dtype != np.uint8:
            norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            current = norm.astype(np.uint8)
        else:
            current = image.copy()

        for step in self.steps:
            if not step.enabled:
                continue
            current = self._apply_step(current, step)

        return current

    def _apply_step(self, img: np.ndarray, step: FilterStep) -> np.ndarray:
        name = step.name.lower()
        p = step.params

        if name in self.custom_filters:
            return self.custom_filters[name](img, p)

        if name == "blur":
            k = int(p.get("ksize", 5))
            k = max(1, k if k % 2 == 1 else k + 1)
            return cv2.blur(img, (k, k))

        elif name in ("gaussian_blur", "gss_blur"):
            k = int(p.get("ksize", 5))
            k = max(1, k if k % 2 == 1 else k + 1)
            sigma = float(p.get("sigma", 0.0))
            return cv2.GaussianBlur(img, (k, k), sigma)

        elif name == "median_blur":
            k = int(p.get("ksize", 5))
            k = max(1, k if k % 2 == 1 else k + 1)
            return cv2.medianBlur(img, k)

        elif name == "sharpen":
            # 3x3 unsharp mask / Laplacian sharpening
            strength = float(p.get("strength", 1.0))
            kernel = np.array([
                [0, -1, 0],
                [-1, 4 + strength, -1],
                [0, -1, 0]
            ], dtype=np.float32)
            filtered = cv2.filter2D(img.astype(np.float32), -1, kernel)
            return np.clip(filtered, 0, 255).astype(np.uint8)

        elif name == "canny":
            low = float(p.get("lower", 50))
            high = float(p.get("upper", 150))
            return cv2.Canny(img, low, high)

        elif name == "sobel":
            k = int(p.get("ksize", 3))
            k = max(1, k if k % 2 == 1 else k + 1)
            grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=k)
            grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=k)
            mag = cv2.magnitude(grad_x, grad_y)
            return np.clip(mag, 0, 255).astype(np.uint8)

        elif name in ("threshold", "dual_threshold"):
            method = p.get("method", "dual")
            if method == "otsu":
                _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return thresh
            elif method == "adaptive":
                block = int(p.get("block_size", 11))
                block = max(3, block if block % 2 == 1 else block + 1)
                c_val = float(p.get("c_val", 2.0))
                return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c_val)
            else:
                # Dual threshold range [lower, upper]
                low = int(p.get("lower", 128))
                high = int(p.get("upper", 255))
                mask = cv2.inRange(img, low, high)
                return mask

        elif name in ("morph_open", "open"):
            kernel = self._get_morph_kernel(p)
            iters = int(p.get("iterations", 1))
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iters)

        elif name in ("morph_close", "close"):
            kernel = self._get_morph_kernel(p)
            iters = int(p.get("iterations", 1))
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iters)

        elif name in ("morph_gradient", "gradient"):
            kernel = self._get_morph_kernel(p)
            iters = int(p.get("iterations", 1))
            return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=iters)

        return img

    def _get_morph_kernel(self, p: Dict[str, Any]) -> np.ndarray:
        k = int(p.get("ksize", 3))
        shape_str = p.get("shape", "rect").lower()
        shape = cv2.MORPH_RECT
        if shape_str in ("ellipse", "circle"):
            shape = cv2.MORPH_ELLIPSE
        elif shape_str == "cross":
            shape = cv2.MORPH_CROSS
        return cv2.getStructuringElement(shape, (k, k))

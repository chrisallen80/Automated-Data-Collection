"""
Global reactive UI state for the browser-based Automated Data Collection GUI.
"""

from __future__ import annotations
import base64
import io
from typing import Optional, List, Dict, Any
import numpy as np
import cv2

from adc.acquisition.engine import AcquisitionEngine, AcquisitionState
from adc.imaging.clustering import Cluster
from adc.imaging.grid_planner import ScanTile


class UIState:
    """
    Holds UI session state and provides helper methods to convert numpy images
    into browser-compatible Base64 PNG/JPEG streams.
    """

    def __init__(self, engine: AcquisitionEngine):
        self.engine = engine
        self.active_tab: str = "survey"

        # Canvas overlays
        self.show_clusters: bool = True
        self.show_tiles: bool = True

        # Selected cluster IDs for scanning
        self.selected_cluster_ids: set[int] = set()

        # UI parameters
        self.survey_size: int = 512
        self.tile_size: int = 256
        self.overlap_percent: int = 10
        self.tighten_boxes: bool = True
        self.scan_pattern: str = "snake"
        self.segmentation_method: str = "contours"

        # Virtual detector parameters
        self.vadf_inner_r: float = 30.0
        self.vadf_outer_r: float = 80.0
        self.vbf_radius: float = 25.0

        # Arbitrary in-flight correction position
        self.arbitrary_corr_px: Optional[Tuple[int, int]] = None
        self.arbitrary_corr_nm: Optional[Tuple[float, float]] = None

    @staticmethod
    def numpy_to_base64_png(image: np.ndarray) -> str:
        """Convert a 2D or 3D numpy array into a base64 encoded PNG data URI."""
        if image is None or image.size == 0:
            return ""

        # Normalize to uint8 if needed
        if image.dtype != np.uint8:
            norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            uint8_img = norm.astype(np.uint8)
        else:
            uint8_img = image

        success, encoded = cv2.imencode(".png", uint8_img)
        if not success:
            return ""
        b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    def render_survey_with_overlays(self) -> str:
        """
        Renders the current survey image with cluster bounding boxes and planned scan tiles.
        """
        if self.engine.raw_survey_image is None:
            return ""

        base = self.engine.raw_survey_image.copy()
        if len(base.shape) == 2:
            rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
        else:
            rgb = base.copy()

        # Draw detected clusters
        if self.show_clusters:
            for c in self.engine.detected_clusters:
                x, y, w, h = c.bbox
                is_selected = (c.id in self.selected_cluster_ids) if self.selected_cluster_ids else c.selected
                color = (0, 255, 120) if is_selected else (120, 120, 120) # Bright green if selected
                cv2.rectangle(rgb, (x, y), (x + w, y + h), color, 2)
                # Centroid marker
                cx, cy = int(c.center[0]), int(c.center[1])
                cv2.circle(rgb, (cx, cy), 3, (0, 200, 255), -1)
                # Label
                cv2.putText(
                    rgb, f"#{c.id}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
                )

        # Draw planned scan tiles (with snake order gradient)
        if self.show_tiles and self.engine.planned_tiles:
            num_tiles = len(self.engine.planned_tiles)
            for idx, t in enumerate(self.engine.planned_tiles):
                # Completed tiles in gold, current tile in cyan, pending in magenta
                if t.completed:
                    t_color = (255, 200, 0) # Gold
                elif idx == self.engine.current_tile_index and self.engine.state == AcquisitionState.RUNNING:
                    t_color = (0, 255, 255) # Cyan
                else:
                    t_color = (255, 60, 160) # Magenta

                cv2.rectangle(rgb, (t.x, t.y), (t.x + t.width, t.y + t.height), t_color, 1)
                cv2.putText(
                    rgb, f"T{idx+1}", (t.x + 4, t.y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, t_color, 1, cv2.LINE_AA
                )

        # Draw arbitrary in-flight correction target if set
        if self.arbitrary_corr_px is not None:
            px, py = self.arbitrary_corr_px
            target_color = (0, 165, 255) # Bright Amber / Orange
            # Crosshair circle target
            cv2.circle(rgb, (px, py), 10, target_color, 1, cv2.LINE_AA)
            cv2.line(rgb, (px - 16, py), (px + 16, py), target_color, 1, cv2.LINE_AA)
            cv2.line(rgb, (px, py - 16), (px, py + 16), target_color, 1, cv2.LINE_AA)
            cv2.putText(
                rgb, "CORR TARGET", (px + 12, max(12, py - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, target_color, 1, cv2.LINE_AA
            )

        return self.numpy_to_base64_png(rgb)

    def render_diffraction_with_apertures(self) -> str:
        """
        Renders the active 4D-STEM diffraction pattern with virtual aperture rings overlaid.
        """
        frame = self.engine.latest_diffraction_frame
        if frame is None and self.engine.camera is not None:
            frame = self.engine.camera.get_latest_frame()

        if frame is None:
            # Generate blank 256x256 placeholder
            frame = np.zeros((256, 256), dtype=np.uint8)

        # Normalize and colorize
        norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_map = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)

        h, w = norm.shape[:2]
        cx, cy = w // 2, h // 2

        # Draw VBF aperture (cyan circle)
        cv2.circle(color_map, (cx, cy), int(self.vbf_radius), (255, 255, 0), 1)

        # Draw VADF annular apertures (red circles)
        cv2.circle(color_map, (cx, cy), int(self.vadf_inner_r), (0, 0, 255), 1)
        cv2.circle(color_map, (cx, cy), int(self.vadf_outer_r), (0, 0, 255), 1)

        return self.numpy_to_base64_png(color_map)

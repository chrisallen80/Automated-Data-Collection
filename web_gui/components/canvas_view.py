"""
Micrograph Canvas View Component.
Displays raw survey micrograph, filter pipeline preview, and overlay annotations.
"""

from __future__ import annotations
from typing import Callable
from nicegui import ui
from ..state import UIState


class CanvasView:
    def __init__(self, state: UIState, on_refresh: Optional[Callable[[], None]] = None):
        self.state = state
        self.on_refresh = on_refresh
        self.image_element: Optional[ui.image] = None
        self.view_mode: str = "overlay"  # 'overlay', 'raw', 'processed'

    def build(self) -> ui.card:
        with ui.card().classes("w-full h-full p-4 bg-slate-900 border border-slate-800 rounded-xl shadow-lg"):
            with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-800"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("photo_size_select_actual", size="sm", color="primary")
                    ui.label("Micrograph Survey Viewer").classes("text-lg font-semibold text-slate-100")

                with ui.row().classes("items-center gap-3"):
                    self.mode_toggle = ui.toggle(
                        {"overlay": "Annotations", "raw": "Raw STEM", "processed": "Binary Mask"},
                        value=self.view_mode,
                        on_change=self._on_mode_change
                    ).props("dense no-caps rounded unelevated color=primary text-color=white")

                    with ui.row().classes("items-center gap-2"):
                        ui.checkbox("Clusters", value=self.state.show_clusters, on_change=self._toggle_clusters).props("dense color=positive")
                        ui.checkbox("Tiles", value=self.state.show_tiles, on_change=self._toggle_tiles).props("dense color=secondary")

            with ui.row().classes("w-full justify-center items-center min-h-[420px] bg-black/60 rounded-lg overflow-hidden border border-slate-800/80 my-2"):
                self.image_element = ui.interactive_image(
                    self._get_current_image_src(),
                    on_mouse=self._on_mouse,
                    events=["mousedown"],
                    cross=True
                ).classes("max-h-[500px] object-contain rounded cursor-crosshair")

            with ui.row().classes("w-full justify-between items-center text-xs text-slate-400 pt-1"):
                img = self.state.engine.raw_survey_image
                shape_text = f"{img.shape[1]} x {img.shape[0]} px" if img is not None else "No image loaded"
                clusters_text = f"{len(self.state.engine.detected_clusters)} features"
                tiles_text = f"{len(self.state.engine.planned_tiles)} tiles queued"
                ui.label(f"Dimensions: {shape_text}")
                ui.label(f"Objects: {clusters_text} | {tiles_text}")
                with ui.row().classes("items-center gap-2"):
                    if self.state.arbitrary_corr_px:
                        ui.label(f"Corr Target: {self.state.arbitrary_corr_px}").classes("text-amber-400 font-mono")
                        ui.button(icon="clear", on_click=self._clear_arbitrary_point).props("flat dense round size=xs color=negative").tooltip("Clear target")
                    ui.label("Click survey to set Arbitrary Correction Target").classes("text-slate-500 italic")

        return self

    def _on_mouse(self, e):
        if e.type == "mousedown" and e.image_x is not None and e.image_y is not None:
            img = self.state.engine.raw_survey_image
            if img is None:
                return

            px_x = int(round(e.image_x))
            px_y = int(round(e.image_y))

            # Clamp within image bounds
            px_x = max(0, min(img.shape[1] - 1, px_x))
            px_y = max(0, min(img.shape[0] - 1, px_y))

            self.state.arbitrary_corr_px = (px_x, px_y)

            # Calculate corresponding stage coordinates
            curr_pos = self.state.engine.microscope.get_stage_position()
            nm_per_px = self.state.engine.planner.nm_per_pixel
            x_nm = curr_pos.x_nm + (px_x - img.shape[1] / 2) * nm_per_px
            y_nm = curr_pos.y_nm + (px_y - img.shape[0] / 2) * nm_per_px
            self.state.arbitrary_corr_nm = (x_nm, y_nm)

            self.state.engine.set_arbitrary_correction_position(x_nm=x_nm, y_nm=y_nm, px_x=px_x, px_y=px_y)
            ui.notify(f"Correction Target set to ({px_x}, {px_y}) px | ({x_nm:.1f}, {y_nm:.1f}) nm", type="positive")
            self.update()
            if self.on_refresh:
                self.on_refresh()

    def _clear_arbitrary_point(self):
        self.state.arbitrary_corr_px = None
        self.state.arbitrary_corr_nm = None
        self.state.engine.set_arbitrary_correction_position(x_nm=None, y_nm=None, px_x=None, px_y=None)
        ui.notify("Cleared arbitrary correction target", type="info")
        self.update()
        if self.on_refresh:
            self.on_refresh()

    def _get_current_image_src(self) -> str:
        if self.view_mode == "raw":
            return self.state.numpy_to_base64_png(self.state.engine.raw_survey_image)
        elif self.view_mode == "processed":
            return self.state.numpy_to_base64_png(self.state.engine.processed_survey_image)
        else:
            return self.state.render_survey_with_overlays()

    def _on_mode_change(self, e):
        self.view_mode = e.value
        self.update()

    def _toggle_clusters(self, e):
        self.state.show_clusters = e.value
        self.update()

    def _toggle_tiles(self, e):
        self.state.show_tiles = e.value
        self.update()

    def update(self) -> None:
        if self.image_element:
            src = self._get_current_image_src()
            if src:
                self.image_element.set_source(src)

"""
Filter Pipeline Configuration Panel.
Interactive controls for image preprocessing, smoothing, thresholding, and morphology.
"""

from __future__ import annotations
from typing import Callable, Optional
from nicegui import ui
from adc.imaging.pipeline import FilterStep
from ..state import UIState


class PipelinePanel:
    def __init__(self, state: UIState, on_pipeline_updated: Optional[Callable[[], None]] = None):
        self.state = state
        self.on_pipeline_updated = on_pipeline_updated

    def build(self) -> ui.card:
        with ui.card().classes("w-full p-4 bg-slate-900 border border-slate-800 rounded-xl shadow-lg"):
            with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-800"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("tune", size="sm", color="primary")
                    ui.label("Preprocessing Filter Pipeline").classes("text-lg font-semibold text-slate-100")

                ui.button("Process Survey", icon="play_arrow", on_click=self._run_pipeline).props("unelevated color=primary")

            with ui.grid(columns=2).classes("w-full gap-4 pt-3"):
                # Column 1: Smoothing & Sharpening
                with ui.column().classes("gap-3 bg-slate-950/50 p-3 rounded-lg border border-slate-800/60"):
                    ui.label("1. Smoothing & Contrast").classes("text-sm font-medium text-slate-300")

                    # Gaussian blur
                    with ui.row().classes("w-full items-center justify-between"):
                        self.gss_enable = ui.checkbox("Gaussian Blur", value=True, on_change=self._on_param_change).props("dense color=primary")
                        ui.label("Kernel").classes("text-xs text-slate-400")
                    self.gss_ksize = ui.slider(min=1, max=15, step=2, value=5, on_change=self._on_param_change).props("label label-always color=primary")

                    # Sharpen
                    with ui.row().classes("w-full items-center justify-between pt-2"):
                        self.sharp_enable = ui.checkbox("Sharpen Filter", value=False, on_change=self._on_param_change).props("dense color=primary")
                        ui.label("Strength").classes("text-xs text-slate-400")
                    self.sharp_strength = ui.slider(min=0.2, max=3.0, step=0.1, value=1.0, on_change=self._on_param_change).props("label label-always color=primary")

                # Column 2: Thresholding & Morphology
                with ui.column().classes("gap-3 bg-slate-950/50 p-3 rounded-lg border border-slate-800/60"):
                    ui.label("2. Binarization & Morphology").classes("text-sm font-medium text-slate-300")

                    # Threshold mode
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Threshold Method").classes("text-xs text-slate-400")
                        thresh_opts = ["dual", "otsu", "adaptive"] + list(self.state.engine.pipeline.custom_filters.keys())
                        self.thresh_method = ui.select(
                            thresh_opts, value="dual", on_change=self._on_thresh_method_change
                        ).props("dense outlined options-dense")

                    # Dual threshold sliders
                    with ui.row().classes("w-full items-center justify-between") as self.thresh_lower_row:
                        ui.label("Lower Cutoff").classes("text-xs text-slate-400")
                        self.thresh_lower_lbl = ui.label("120").classes("text-xs font-mono text-primary")
                    self.thresh_lower = ui.slider(min=0, max=255, value=120, on_change=self._on_param_change).props("label color=primary")

                    with ui.row().classes("w-full items-center justify-between") as self.thresh_upper_row:
                        ui.label("Upper Cutoff").classes("text-xs text-slate-400")
                        self.thresh_upper_lbl = ui.label("255").classes("text-xs font-mono text-primary")
                    self.thresh_upper = ui.slider(min=0, max=255, value=255, on_change=self._on_param_change).props("label color=primary")

                    # Morphological opening/closing
                    with ui.row().classes("w-full items-center justify-between pt-2"):
                        self.morph_enable = ui.checkbox("Morphological Cleanup (Open/Close)", value=True, on_change=self._on_param_change).props("dense color=primary")

            # Active Custom Plugin Filters list
            with ui.column().classes("w-full mt-3 p-3 bg-slate-950/60 rounded-lg border border-slate-800/80 gap-2") as self.custom_filters_container:
                self._render_custom_filters_ui()

        return self

    def _render_custom_filters_ui(self):
        if not hasattr(self, "custom_filters_container") or self.custom_filters_container is None:
            return
        self.custom_filters_container.clear()
        with self.custom_filters_container:
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Active Pipeline Filters & Custom Plugins").classes("text-xs font-semibold text-slate-300")
                ui.button("Refresh Plugins", icon="refresh", on_click=self.refresh_plugin_options).props("dense flat size=xs color=primary")

            custom_keys = list(self.state.engine.pipeline.custom_filters.keys())
            if not custom_keys:
                ui.label("No external filter plugins loaded. (Load .py scripts in the Microscope & Plugins tab)").classes("text-xs text-slate-500 italic")
            else:
                for k in custom_keys:
                    with ui.row().classes("w-full items-center justify-between bg-slate-900/60 px-2 py-1 rounded border border-slate-800"):
                        ui.label(f"• Plugin: {k}").classes("text-xs font-mono text-cyan-400")
                        ui.button("Select Method", on_click=lambda name=k: self._select_custom_method(name)).props("dense flat size=xs color=cyan")

    def _select_custom_method(self, name: str):
        self.thresh_method.set_value(name)
        self._on_thresh_method_change(None)

    def refresh_plugin_options(self):
        """Update select options when external plugins are loaded."""
        thresh_opts = ["dual", "otsu", "adaptive"] + list(self.state.engine.pipeline.custom_filters.keys())
        self.thresh_method.options = thresh_opts
        self.thresh_method.update()
        self._render_custom_filters_ui()

    def _on_thresh_method_change(self, e):
        val = self.thresh_method.value
        is_dual = (val == "dual")
        self.thresh_lower_row.set_visibility(is_dual)
        self.thresh_lower.set_visibility(is_dual)
        self.thresh_upper_row.set_visibility(is_dual)
        self.thresh_upper.set_visibility(is_dual)
        self._on_param_change(e)

    def _sync_pipeline_params(self) -> None:
        """Update engine's ImagePipeline with current GUI widget values."""
        steps = []
        # Gaussian blur
        steps.append(FilterStep(
            name="gaussian_blur",
            enabled=self.gss_enable.value,
            params={"ksize": int(self.gss_ksize.value), "sigma": float(self.gss_ksize.value / 3.0)}
        ))
        # Sharpen
        steps.append(FilterStep(
            name="sharpen",
            enabled=self.sharp_enable.value,
            params={"strength": float(self.sharp_strength.value)}
        ))
        # Threshold
        method = self.thresh_method.value
        if method in self.state.engine.pipeline.custom_filters:
            # User selected a custom threshold plugin
            steps.append(FilterStep(
                name=method,
                enabled=True,
                params={}
            ))
        else:
            steps.append(FilterStep(
                name="threshold",
                enabled=True,
                params={
                    "method": method,
                    "lower": int(self.thresh_lower.value),
                    "upper": int(self.thresh_upper.value)
                }
            ))
        # Morphology
        if self.morph_enable.value:
            steps.append(FilterStep(name="morph_open", enabled=True, params={"ksize": 3, "iterations": 1}))
            steps.append(FilterStep(name="morph_close", enabled=True, params={"ksize": 3, "iterations": 1}))

        self.state.engine.pipeline.steps = steps

    def _on_param_change(self, e):
        self.thresh_lower_lbl.set_text(str(int(self.thresh_lower.value)))
        self.thresh_upper_lbl.set_text(str(int(self.thresh_upper.value)))
        self._sync_pipeline_params()
        if self.state.engine.raw_survey_image is not None:
            self._run_pipeline()

    def _run_pipeline(self) -> None:
        self._sync_pipeline_params()
        if self.state.engine.raw_survey_image is None:
            ui.notify("No survey image available. Please acquire or load a survey first.", type="warning")
            return

        self.state.engine.process_survey()
        if self.on_pipeline_updated:
            self.on_pipeline_updated()
        ui.notify("Pipeline executed successfully.", type="positive")

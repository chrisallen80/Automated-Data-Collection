"""
Scan Grid Planner and Acquisition Queue Control Panel.
"""

from __future__ import annotations
from typing import Callable, Optional
from nicegui import ui
from ..state import UIState
from adc.acquisition.engine import AcquisitionState


class ScanPanel:
    def __init__(self, state: UIState, on_plan_updated: Optional[Callable[[], None]] = None):
        self.state = state
        self.on_plan_updated = on_plan_updated
        self.progress_bar: Optional[ui.linear_progress] = None
        self.status_label: Optional[ui.label] = None
        self.progress_text: Optional[ui.label] = None

    def build(self) -> ui.card:
        with ui.card().classes("w-full p-4 bg-slate-900 border border-slate-800 rounded-xl shadow-lg"):
            with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-800"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("grid_4x4", size="sm", color="secondary")
                    ui.label("Scan Grid Planning & Acquisition Queue").classes("text-lg font-semibold text-slate-100")

                with ui.row().classes("gap-2"):
                    ui.button("Plan Scan Tiles", icon="account_tree", on_click=self._plan_tiles).props("unelevated color=secondary")

            # Grid parameters
            with ui.row().classes("w-full gap-4 items-center bg-slate-950/40 p-3 rounded-lg border border-slate-800/60 my-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("Tile Size:").classes("text-xs text-slate-400")
                    self.tile_size_select = ui.select(
                        [128, 256, 512], value=256, on_change=self._on_grid_param_change
                    ).props("dense outlined options-dense")

                with ui.row().classes("items-center gap-2"):
                    ui.label("Overlap (%):").classes("text-xs text-slate-400")
                    self.overlap_slider = ui.slider(min=0, max=50, step=5, value=10, on_change=self._on_grid_param_change).props("label color=secondary").classes("w-32")

                with ui.row().classes("items-center gap-2"):
                    ui.label("Trajectory:").classes("text-xs text-slate-400")
                    self.trajectory_select = ui.select(
                        ["snake", "raster", "spiral"], value="snake"
                    ).props("dense outlined options-dense")

                self.tighten_cb = ui.checkbox("Tighten Bounding Boxes", value=True, on_change=self._on_grid_param_change).props("dense color=secondary")

            # In-Flight Corrections quick settings
            with ui.row().classes("w-full gap-4 items-center bg-slate-950/40 p-3 rounded-lg border border-slate-800/60 my-2"):
                ui.icon("tune", size="xs", color="warning")
                ui.label("In-Flight Corrections:").classes("text-xs font-semibold text-slate-300")

                # Drift quick toggle & interval
                self.drift_cb = ui.checkbox(
                    "Drift Corr",
                    value=self.state.engine.drift_routine.enabled,
                    on_change=lambda e: setattr(self.state.engine.drift_routine, "enabled", bool(e.value))
                ).props("dense color=warning")
                self.drift_interval = ui.number(
                    value=self.state.engine.drift_routine.interval_scans,
                    min=1, max=100, step=1, prefix="Every", suffix="scans",
                    on_change=lambda e: setattr(self.state.engine.drift_routine, "interval_scans", max(1, int(e.value or 1)))
                ).props("dense outlined").classes("w-32")

                # Focus quick toggle & interval
                self.focus_cb = ui.checkbox(
                    "Autofocus",
                    value=self.state.engine.autofocus_routine.enabled,
                    on_change=lambda e: setattr(self.state.engine.autofocus_routine, "enabled", bool(e.value))
                ).props("dense color=cyan")
                self.focus_interval = ui.number(
                    value=self.state.engine.autofocus_routine.interval_scans,
                    min=1, max=100, step=1, prefix="Every", suffix="scans",
                    on_change=lambda e: setattr(self.state.engine.autofocus_routine, "interval_scans", max(1, int(e.value or 1)))
                ).props("dense outlined").classes("w-32")

            # Execution controls and progress section
            with ui.column().classes("w-full bg-slate-950/60 p-4 rounded-lg border border-slate-800 mt-2 gap-3"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("gap-2 items-center"):
                        self.btn_start = ui.button("Start Acquisition", icon="play_arrow", on_click=self._start_acquisition).props("unelevated color=positive")
                        self.btn_pause = ui.button("Pause", icon="pause", on_click=self._pause_acquisition).props("flat color=warning")
                        self.btn_resume = ui.button("Resume", icon="play_arrow", on_click=self._resume_acquisition).props("flat color=positive")
                        self.btn_stop = ui.button("Stop", icon="stop", on_click=self._stop_acquisition).props("flat color=negative")
                        self.btn_resume.set_visibility(False)

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Status:").classes("text-xs text-slate-400")
                        self.status_label = ui.label("IDLE").classes("text-xs font-mono font-bold text-slate-200 bg-slate-800 px-2 py-1 rounded")

                # Progress bar
                self.progress_bar = ui.linear_progress(value=0.0, show_value=False).props("color=secondary size=10px rounded")

                with ui.row().classes("w-full justify-between items-center text-xs text-slate-400"):
                    self.progress_text = ui.label("Queue: 0/0 tiles (0%)").classes("font-mono")
                    self.eta_text = ui.label("Ready to run").classes("font-mono")

        # Hook engine callbacks to update UI
        self.state.engine.subscribe("on_tile_progress", self._on_engine_tile_progress)
        self.state.engine.subscribe("on_state_change", self._on_engine_state_change)

        return self

    def _on_grid_param_change(self, e):
        pass

    def _plan_tiles(self):
        tile_size = int(self.tile_size_select.value)
        overlap = float(self.overlap_slider.value) / 100.0
        tighten = bool(self.tighten_cb.value)

        tiles = self.state.engine.plan_scan_tiles(tile_size=tile_size, overlap=overlap, tighten=tighten)
        self.progress_text.set_text(f"Queue: 0/{len(tiles)} tiles (0%)")
        self.progress_bar.set_value(0.0)

        if self.on_plan_updated:
            self.on_plan_updated()

        ui.notify(f"Planned {len(tiles)} scan tiles.", type="info")

    def _start_acquisition(self):
        try:
            if not self.state.engine.planned_tiles:
                self._plan_tiles()
            self.state.engine.start_acquisition()
            self.btn_start.disable()
            self.btn_resume.set_visibility(False)
            ui.notify("Acquisition started.", type="positive")
        except Exception as e:
            ui.notify(str(e), type="negative")

    def _pause_acquisition(self):
        self.state.engine.pause_acquisition()
        self.btn_resume.set_visibility(True)
        ui.notify("Acquisition paused.", type="warning")

    def _resume_acquisition(self):
        self.state.engine.resume_acquisition()
        self.btn_resume.set_visibility(False)
        ui.notify("Acquisition resumed.", type="positive")

    def _stop_acquisition(self):
        self.state.engine.stop_acquisition()
        self.btn_start.enable()
        self.btn_resume.set_visibility(False)
        ui.notify("Acquisition halted.", type="negative")

    def _on_engine_tile_progress(self, current: int, total: int, tile):
        fraction = float(current) / float(total) if total > 0 else 0.0
        if self.progress_bar:
            self.progress_bar.set_value(fraction)
        if self.progress_text:
            self.progress_text.set_text(f"Queue: {current}/{total} tiles ({int(fraction*100)}%)")
        if self.on_plan_updated:
            self.on_plan_updated()

    def _on_engine_state_change(self, state: AcquisitionState, msg: str):
        if self.status_label:
            self.status_label.set_text(state.value)
            if state == AcquisitionState.RUNNING:
                self.status_label.classes("text-secondary", remove="text-slate-200 text-positive text-negative")
            elif state == AcquisitionState.COMPLETED:
                self.status_label.classes("text-positive", remove="text-slate-200 text-secondary text-negative")
                if self.btn_start:
                    self.btn_start.enable()
            elif state in (AcquisitionState.STOPPED, AcquisitionState.ERROR):
                self.status_label.classes("text-negative", remove="text-slate-200 text-secondary text-positive")
                if self.btn_start:
                    self.btn_start.enable()

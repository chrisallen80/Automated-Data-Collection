"""
4D-STEM Live Diffraction and Virtual Detector Component.
"""

from __future__ import annotations
from typing import Optional
from nicegui import ui
import numpy as np

from ..state import UIState
from adc.imaging.stem4d import STEM4DProcessor


class LiveSTEM4DView:
    def __init__(self, state: UIState):
        self.state = state
        self.diffraction_img_el: Optional[ui.image] = None
        self.com_label: Optional[ui.label] = None
        self.cryst_label: Optional[ui.label] = None
        self.pbr_label: Optional[ui.label] = None

    def build(self) -> ui.card:
        with ui.card().classes("w-full p-4 bg-slate-900 border border-slate-800 rounded-xl shadow-lg"):
            with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-800"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("lens", size="sm", color="accent")
                    ui.label("4D-STEM Diffraction & Virtual Detectors").classes("text-lg font-semibold text-slate-100")

                with ui.row().classes("gap-2"):
                    ui.button("Capture Frame", icon="camera", on_click=self._capture_frame).props("flat dense color=accent")

            with ui.grid(columns=2).classes("w-full gap-4 pt-3"):
                # Left: Live diffraction pattern viewer
                with ui.column().classes("w-full items-center bg-black/70 p-3 rounded-lg border border-slate-800/80"):
                    self.diffraction_img_el = ui.image(self.state.render_diffraction_with_apertures()).classes("max-h-[340px] object-contain rounded")
                    with ui.row().classes("w-full justify-between items-center text-xs text-slate-400 pt-2"):
                        ui.label("Detector: Merlin Medipix (256x256)")
                        ui.label("Overlay: Cyan (VBF) | Red (VADF)")

                # Right: Virtual Apertures and Live Analysis Metrics
                with ui.column().classes("gap-3 w-full"):
                    with ui.column().classes("w-full bg-slate-950/50 p-3 rounded-lg border border-slate-800/60 gap-2"):
                        ui.label("Virtual Aperture Radii (pixels)").classes("text-sm font-medium text-slate-300")

                        # VBF
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("VBF Disc Radius").classes("text-xs text-slate-400")
                            self.vbf_lbl = ui.label("25 px").classes("text-xs font-mono text-cyan-400")
                        self.vbf_slider = ui.slider(min=5, max=100, value=25, on_change=self._on_aperture_change).props("color=cyan")

                        # VADF Inner
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("VADF Inner Radius").classes("text-xs text-slate-400")
                            self.vadf_in_lbl = ui.label("30 px").classes("text-xs font-mono text-red-400")
                        self.vadf_in_slider = ui.slider(min=10, max=110, value=30, on_change=self._on_aperture_change).props("color=red")

                        # VADF Outer
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("VADF Outer Radius").classes("text-xs text-slate-400")
                            self.vadf_out_lbl = ui.label("80 px").classes("text-xs font-mono text-red-400")
                        self.vadf_out_slider = ui.slider(min=30, max=125, value=80, on_change=self._on_aperture_change).props("color=red")

                    # Live Analysis Metrics
                    with ui.column().classes("w-full bg-slate-950/50 p-3 rounded-lg border border-slate-800/60 gap-2"):
                        ui.label("Live In-Flight Analysis").classes("text-sm font-medium text-slate-300")

                        with ui.row().classes("w-full items-center justify-between text-xs"):
                            ui.label("Center of Mass (DPC)").classes("text-slate-400")
                            self.com_label = ui.label("Δx: 0.0, Δy: 0.0").classes("font-mono text-slate-200")

                        with ui.row().classes("w-full items-center justify-between text-xs"):
                            ui.label("Crystallinity State").classes("text-slate-400")
                            self.cryst_label = ui.label("Amorphous / Carbon").classes("font-mono font-semibold text-warning")

                        with ui.row().classes("w-full items-center justify-between text-xs"):
                            ui.label("Peak-to-Background Ratio").classes("text-slate-400")
                            self.pbr_label = ui.label("1.0x").classes("font-mono text-slate-200")

        # Subscribe to new frames from engine
        self.state.engine.subscribe("on_frame_received", self._on_new_frame)
        return self

    def _on_aperture_change(self, e):
        self.state.vbf_radius = float(self.vbf_slider.value)
        self.state.vadf_inner_r = float(self.vadf_in_slider.value)
        self.state.vadf_outer_r = float(self.vadf_out_slider.value)

        self.vbf_lbl.set_text(f"{int(self.state.vbf_radius)} px")
        self.vadf_in_lbl.set_text(f"{int(self.state.vadf_inner_r)} px")
        self.vadf_out_lbl.set_text(f"{int(self.state.vadf_outer_r)} px")

        self.update_diffraction()

    def _capture_frame(self):
        if self.state.engine.camera:
            frame = self.state.engine.camera.get_latest_frame()
            if frame is not None:
                self.state.latest_diffraction_frame = frame
                self._analyze_frame(frame)
                self.update_diffraction()
                ui.notify("Diffraction pattern updated.", type="positive")

    def _on_new_frame(self, frame: np.ndarray):
        self._analyze_frame(frame)
        self.update_diffraction()

    def _analyze_frame(self, frame: np.ndarray):
        # COM
        com_x, com_y = STEM4DProcessor.compute_center_of_mass(frame)
        if self.com_label:
            self.com_label.set_text(f"Δx: {com_x:+.2f}, Δy: {com_y:+.2f}")

        # Bragg Crystallinity
        cryst = STEM4DProcessor.compute_crystallinity_index(frame)
        if self.cryst_label and self.pbr_label:
            self.pbr_label.set_text(f"{cryst['peak_to_background_ratio']:.1f}x")
            if cryst["is_crystalline"]:
                self.cryst_label.set_text("CRYSTALLINE (Bragg peaks)")
                self.cryst_label.classes("text-positive", remove="text-warning text-negative")
            else:
                self.cryst_label.set_text("Amorphous / Carbon")
                self.cryst_label.classes("text-warning", remove="text-positive text-negative")

    def update_diffraction(self):
        if self.diffraction_img_el:
            src = self.state.render_diffraction_with_apertures()
            if src:
                self.diffraction_img_el.set_source(src)

"""
Microscope Hardware Status, In-Flight Corrections, and Plugin Loader Component.
Provides granular, independent control over drift tracking, autofocus routines,
scan intervals, and reference location strategies (Previous Scan, Next Scan, or Fixed Region).
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from pathlib import Path
from nicegui import ui

from ..state import UIState
from adc.plugins import GLOBAL_REGISTRY
from adc.corrections import CorrectionLocationMode, BaseCorrectionRoutine


class CorrectionsView:
    def __init__(self, state: UIState):
        self.state = state
        self.stage_label: Optional[ui.label] = None
        self.defocus_label: Optional[ui.label] = None
        self.drift_label: Optional[ui.label] = None
        self.emission_label: Optional[ui.label] = None
        self.active_corrections_container: Optional[ui.column] = None

    def build(self) -> ui.card:
        engine = self.state.engine

        with ui.card().classes("w-full p-4 bg-slate-900 border border-slate-800 rounded-xl shadow-lg"):
            with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-800"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("tune", size="sm", color="warning")
                    ui.label("Hardware Telemetry, In-Flight Corrections & Plugins").classes("text-lg font-semibold text-slate-100")

                with ui.row().classes("gap-2"):
                    ui.button("Refresh Telemetry", icon="sync", on_click=self.refresh_telemetry).props("flat dense color=warning")

            with ui.grid(columns=3).classes("w-full gap-4 pt-3"):
                # ==========================================
                # Column 1: Stage & Lens Telemetry
                # ==========================================
                with ui.column().classes("w-full bg-slate-950/50 p-3 rounded-lg border border-slate-800/60 gap-2"):
                    ui.label("Microscope Column State").classes("text-sm font-medium text-slate-300")

                    self.stage_label = ui.label("Stage: (0.0, 0.0, 0.0) nm").classes("text-xs font-mono text-slate-200")
                    self.defocus_label = ui.label("Defocus (OLf): 0").classes("text-xs font-mono text-slate-200")
                    self.mag_label = ui.label("Magnification: 50,000x").classes("text-xs font-mono text-slate-200")
                    self.ht_label = ui.label("HT: 200.0 kV").classes("text-xs font-mono text-slate-200")

                    # Beam Blank toggle
                    with ui.row().classes("w-full items-center justify-between pt-2 border-t border-slate-800"):
                        ui.label("Beam Blank").classes("text-xs text-slate-300")
                        self.blank_toggle = ui.switch(value=False, on_change=self._toggle_blank).props("color=negative")

                    # Stage jog buttons
                    ui.label("Stage Jog (nm)").classes("text-xs text-slate-400 pt-2")
                    with ui.row().classes("w-full justify-center gap-1"):
                        ui.button("↑ +Y", on_click=lambda: self._jog_stage(0, 500)).props("dense flat outline size=xs")
                    with ui.row().classes("w-full justify-center gap-1"):
                        ui.button("← -X", on_click=lambda: self._jog_stage(-500, 0)).props("dense flat outline size=xs")
                        ui.button("→ +X", on_click=lambda: self._jog_stage(500, 0)).props("dense flat outline size=xs")
                    with ui.row().classes("w-full justify-center gap-1"):
                        ui.button("↓ -Y", on_click=lambda: self._jog_stage(0, -500)).props("dense flat outline size=xs")

                    # Gun emission
                    with ui.row().classes("w-full items-center justify-between pt-2 border-t border-slate-800"):
                        ui.label("Cold-FEG Emission").classes("text-xs text-slate-300")
                        self.emission_label = ui.label("Stable (12.5 μA)").classes("text-xs font-mono text-positive")

                # ==========================================
                # Column 2: In-Flight Corrections (Drift & Focus)
                # ==========================================
                with ui.column().classes("w-full bg-slate-950/50 p-3 rounded-lg border border-slate-800/60 gap-3"):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("In-Flight Closed-Loop Corrections").classes("text-sm font-medium text-slate-300")
                        self.active_mode_chip = ui.badge("Both Enabled", color="primary").props("dense")

                    # --- Section A: Drift Correction ---
                    with ui.column().classes("w-full p-2.5 rounded bg-slate-900/80 border border-slate-800 gap-2"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Stage Drift Correction").classes("text-xs font-bold text-amber-400")
                            self.drift_switch = ui.switch(
                                value=engine.drift_routine.enabled,
                                on_change=self._on_drift_enable_change
                            ).props("dense color=warning")

                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label("Frequency:").classes("text-xs text-slate-400")
                            self.drift_interval_input = ui.number(
                                value=engine.drift_routine.interval_scans,
                                min=1,
                                max=100,
                                step=1,
                                prefix="Every",
                                suffix="scans",
                                on_change=self._on_drift_interval_change
                            ).props("dense outlined").classes("w-36")

                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label("Ref Location:").classes("text-xs text-slate-400")
                            self.drift_mode_select = ui.select(
                                {
                                    CorrectionLocationMode.FIXED_REGION.value: "Fixed Region (Cluster ROI)",
                                    CorrectionLocationMode.ARBITRARY_POSITION.value: "Arbitrary Position (Stage / Click)",
                                    CorrectionLocationMode.PREVIOUS_SCAN.value: "Previous Scan Tile",
                                    CorrectionLocationMode.NEXT_SCAN.value: "Next Scan Target"
                                },
                                value=engine.drift_routine.location_mode.value,
                                on_change=self._on_drift_mode_change
                            ).props("dense outlined options-dense").classes("w-52")

                        # Fixed Cluster ID select (shown if FIXED_REGION is selected)
                        with ui.row().classes("w-full items-center gap-2") as self.fixed_cluster_row:
                            ui.label("Anchor Cluster:").classes("text-xs text-slate-400")
                            cluster_options = {
                                c.cluster_id: f"Cluster #{c.cluster_id} ({c.centroid[0]}, {c.centroid[1]})"
                                for c in engine.detected_clusters
                            } if engine.detected_clusters else {0: "Default (First Cluster)"}

                            self.fixed_cluster_select = ui.select(
                                cluster_options,
                                value=engine.drift_routine.fixed_cluster_id or 0,
                                on_change=self._on_fixed_cluster_change
                            ).props("dense outlined options-dense").classes("w-52")

                        with ui.row().classes("w-full items-center justify-between pt-1"):
                            self.drift_label = ui.label("Accumulated: dx=0.0 nm, dy=0.0 nm").classes("text-xs font-mono text-slate-300")
                            ui.button("Reset Drift", on_click=self._reset_drift).props("dense flat outline size=xs color=warning")

                    # --- Section B: Autofocus Correction ---
                    with ui.column().classes("w-full p-2.5 rounded bg-slate-900/80 border border-slate-800 gap-2"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Autofocus Optimization").classes("text-xs font-bold text-cyan-400")
                            self.focus_switch = ui.switch(
                                value=engine.autofocus_routine.enabled,
                                on_change=self._on_focus_enable_change
                            ).props("dense color=cyan")

                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label("Frequency:").classes("text-xs text-slate-400")
                            self.focus_interval_input = ui.number(
                                value=engine.autofocus_routine.interval_scans,
                                min=1,
                                max=100,
                                step=1,
                                prefix="Every",
                                suffix="scans",
                                on_change=self._on_focus_interval_change
                            ).props("dense outlined").classes("w-36")

                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label("Focus Location:").classes("text-xs text-slate-400")
                            self.focus_mode_select = ui.select(
                                {
                                    CorrectionLocationMode.NEXT_SCAN.value: "Next Scan Target",
                                    CorrectionLocationMode.ARBITRARY_POSITION.value: "Arbitrary Position (Stage / Click)",
                                    CorrectionLocationMode.PREVIOUS_SCAN.value: "Previous Scan Tile",
                                    CorrectionLocationMode.FIXED_REGION.value: "Fixed Region (Cluster ROI)"
                                },
                                value=engine.autofocus_routine.location_mode.value,
                                on_change=self._on_focus_mode_change
                            ).props("dense outlined options-dense").classes("w-52")

                        with ui.row().classes("w-full items-center justify-between pt-1"):
                            self.autofocus_res_label = ui.label("Last Focus: optimal").classes("text-xs font-mono text-slate-400")
                            ui.button("Run Focus Sweep", icon="center_focus_strong", on_click=self._run_autofocus).props("dense unelevated size=xs color=cyan")

                    # --- Section C: Arbitrary Reference Target ---
                    with ui.column().classes("w-full p-2.5 rounded bg-slate-900/80 border border-amber-900/60 gap-2") as self.arbitrary_pos_card:
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label("Arbitrary Reference Coordinate").classes("text-xs font-bold text-amber-300")
                            ui.button("Use Current Stage Pos", icon="my_location", on_click=self._use_current_stage_for_arbitrary).props("dense unelevated size=xs color=primary")

                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label("Stage (nm):").classes("text-xs text-slate-400 w-16")
                            self.arb_stage_x = ui.number(label="X (nm)", value=0.0, step=100.0, on_change=self._on_arb_stage_input_change).props("dense outlined").classes("w-24")
                            self.arb_stage_y = ui.number(label="Y (nm)", value=0.0, step=100.0, on_change=self._on_arb_stage_input_change).props("dense outlined").classes("w-24")

                        with ui.row().classes("w-full items-center gap-2"):
                            ui.label("Survey (px):").classes("text-xs text-slate-400 w-16")
                            self.arb_px_x = ui.number(label="Px X", value=256, step=1, on_change=self._on_arb_px_input_change).props("dense outlined").classes("w-24")
                            self.arb_px_y = ui.number(label="Px Y", value=256, step=1, on_change=self._on_arb_px_input_change).props("dense outlined").classes("w-24")
                            ui.button("Clear", icon="close", on_click=self._clear_arbitrary_position).props("dense flat outline size=xs color=negative")

                        ui.label("💡 Tip: Click anywhere on the Survey Micrograph on the left to set this target visually.").classes("text-[11px] text-slate-400 italic")

                    self._update_status_chip()

                # ==========================================
                # Column 3: Pluggable External Routines & Corrections
                # ==========================================
                with ui.column().classes("w-full bg-slate-950/50 p-3 rounded-lg border border-slate-800/60 gap-2"):
                    ui.label("Pluggable Extensions & Routines").classes("text-sm font-medium text-slate-300")
                    ui.label("Load custom Python scripts defining clustering, live feedback, or new correction routines:").classes("text-xs text-slate-400")

                    self.plugin_path_input = ui.input(placeholder="C:/path/to/custom_routine.py").props("dense outlined")
                    ui.button("Load Plugin Script", icon="upload_file", on_click=self._load_plugin).props("dense unelevated color=primary")

                    # Registered Corrections in Engine
                    ui.label("Engine Active Corrections:").classes("text-xs font-medium text-slate-300 pt-2 border-t border-slate-800")
                    self.active_corrections_container = ui.column().classes("w-full text-xs font-mono gap-1 text-slate-300")
                    self._render_engine_corrections()

                    # Discovered general routines
                    ui.label("Available Plugins:").classes("text-xs font-medium text-slate-300 pt-2 border-t border-slate-800")
                    self.routines_list = ui.column().classes("text-xs font-mono text-slate-400 gap-1")
                    self._render_active_routines()

        self.refresh_telemetry()
        return self

    def _update_status_chip(self):
        engine = self.state.engine
        d = engine.drift_routine.enabled
        f = engine.autofocus_routine.enabled

        if d and f:
            self.active_mode_chip.set_text("Both Active (Drift + Focus)")
            self.active_mode_chip.props("color=positive")
        elif d:
            self.active_mode_chip.set_text("Only Drift Correction Active")
            self.active_mode_chip.props("color=warning")
        elif f:
            self.active_mode_chip.set_text("Only Autofocus Active")
            self.active_mode_chip.props("color=cyan")
        else:
            self.active_mode_chip.set_text("Corrections Disabled")
            self.active_mode_chip.props("color=grey")

    def _on_drift_enable_change(self, e):
        self.state.engine.drift_routine.enabled = bool(e.value)
        self._update_status_chip()
        self._render_engine_corrections()
        ui.notify(f"Drift correction {'enabled' if e.value else 'disabled'}.", type="info")

    def _on_drift_interval_change(self, e):
        if e.value is not None:
            self.state.engine.drift_routine.interval_scans = max(1, int(e.value))
            self._render_engine_corrections()

    def _on_drift_mode_change(self, e):
        self.state.engine.drift_routine.location_mode = CorrectionLocationMode(e.value)
        self._render_engine_corrections()
        ui.notify(f"Drift reference set to: {e.value}", type="info")

    def _use_current_stage_for_arbitrary(self):
        try:
            pos = self.state.engine.microscope.get_stage_position()
            x_nm, y_nm = pos.x_nm, pos.y_nm
            self.arb_stage_x.set_value(round(x_nm, 1))
            self.arb_stage_y.set_value(round(y_nm, 1))

            # Convert to pixel if survey image exists
            img = self.state.engine.raw_survey_image
            px_x, px_y = None, None
            if img is not None:
                nm_per_px = self.state.engine.planner.nm_per_pixel
                px_x = int(round(img.shape[1] / 2 + (x_nm - pos.x_nm) / nm_per_px))
                px_y = int(round(img.shape[0] / 2 + (y_nm - pos.y_nm) / nm_per_px))
                px_x = max(0, min(img.shape[1] - 1, px_x))
                px_y = max(0, min(img.shape[0] - 1, px_y))
                self.arb_px_x.set_value(px_x)
                self.arb_px_y.set_value(px_y)
                self.state.arbitrary_corr_px = (px_x, px_y)

            self.state.arbitrary_corr_nm = (x_nm, y_nm)
            self.state.engine.set_arbitrary_correction_position(x_nm=x_nm, y_nm=y_nm, px_x=px_x, px_y=px_y)
            ui.notify(f"Arbitrary target set to stage position: ({x_nm:.1f}, {y_nm:.1f}) nm", type="positive")
        except Exception as err:
            ui.notify(f"Could not read stage: {err}", type="negative")

    def _on_arb_stage_input_change(self, e):
        try:
            x_nm = float(self.arb_stage_x.value or 0.0)
            y_nm = float(self.arb_stage_y.value or 0.0)
            self.state.arbitrary_corr_nm = (x_nm, y_nm)
            self.state.engine.set_arbitrary_correction_position(x_nm=x_nm, y_nm=y_nm)
        except Exception:
            pass

    def _on_arb_px_input_change(self, e):
        try:
            px_x = int(self.arb_px_x.value or 0)
            px_y = int(self.arb_px_y.value or 0)
            self.state.arbitrary_corr_px = (px_x, px_y)
            img = self.state.engine.raw_survey_image
            if img is not None:
                curr_pos = self.state.engine.microscope.get_stage_position()
                nm_per_px = self.state.engine.planner.nm_per_pixel
                x_nm = curr_pos.x_nm + (px_x - img.shape[1] / 2) * nm_per_px
                y_nm = curr_pos.y_nm + (px_y - img.shape[0] / 2) * nm_per_px
                self.arb_stage_x.set_value(round(x_nm, 1))
                self.arb_stage_y.set_value(round(y_nm, 1))
                self.state.arbitrary_corr_nm = (x_nm, y_nm)
                self.state.engine.set_arbitrary_correction_position(x_nm=x_nm, y_nm=y_nm, px_x=px_x, px_y=px_y)
            else:
                self.state.engine.set_arbitrary_correction_position(px_x=px_x, px_y=px_y)
        except Exception:
            pass

    def _clear_arbitrary_position(self):
        self.state.arbitrary_corr_px = None
        self.state.arbitrary_corr_nm = None
        self.state.engine.set_arbitrary_correction_position(x_nm=None, y_nm=None, px_x=None, px_y=None)
        self.arb_stage_x.set_value(0.0)
        self.arb_stage_y.set_value(0.0)
        self.arb_px_x.set_value(0)
        self.arb_px_y.set_value(0)
        ui.notify("Arbitrary target position cleared.", type="info")

    def _on_fixed_cluster_change(self, e):
        self.state.engine.drift_routine.fixed_cluster_id = int(e.value) if e.value is not None else None
        ui.notify(f"Drift anchor set to cluster ID {e.value}", type="info")

    def _reset_drift(self):
        self.state.engine.drift_tracker.reset()
        if self.drift_label:
            self.drift_label.set_text("Accumulated: dx=0.0 nm, dy=0.0 nm")
        ui.notify("Accumulated drift reset to zero.", type="positive")

    def _on_focus_enable_change(self, e):
        self.state.engine.autofocus_routine.enabled = bool(e.value)
        self._update_status_chip()
        self._render_engine_corrections()
        ui.notify(f"Autofocus correction {'enabled' if e.value else 'disabled'}.", type="info")

    def _on_focus_interval_change(self, e):
        if e.value is not None:
            self.state.engine.autofocus_routine.interval_scans = max(1, int(e.value))
            self._render_engine_corrections()

    def _on_focus_mode_change(self, e):
        self.state.engine.autofocus_routine.location_mode = CorrectionLocationMode(e.value)
        self._render_engine_corrections()
        ui.notify(f"Autofocus location set to: {e.value}", type="info")

    def _render_engine_corrections(self):
        if not self.active_corrections_container:
            return
        self.active_corrections_container.clear()
        with self.active_corrections_container:
            for routine in self.state.engine.corrections_manager.all_routines():
                status = "✓ ENABLED" if routine.enabled else "✗ DISABLED"
                color = "text-positive" if routine.enabled else "text-slate-500"
                ui.label(f"• {routine.name.upper()}: [{status}] Every {routine.interval_scans} scans ({routine.location_mode.value})").classes(color)

    def refresh_telemetry(self):
        try:
            params = self.state.engine.microscope.get_all_parameters()
            x = params.get("stage_x_nm", 0.0)
            y = params.get("stage_y_nm", 0.0)
            z = params.get("stage_z_nm", 0.0)
            if self.stage_label:
                self.stage_label.set_text(f"Stage: ({x:.1f}, {y:.1f}, {z:.1f}) nm")
            if self.defocus_label:
                self.defocus_label.set_text(f"Defocus (OLf): {params.get('objective_lens_fine', 0)}")
            if self.mag_label:
                self.mag_label.set_text(f"Magnification: {params.get('magnification', 50000):,.0f}x")
            if self.ht_label:
                self.ht_label.set_text(f"HT: {params.get('high_tension_kv', 200.0):.1f} kV")
        except Exception as e:
            print("Telemetry error:", e)

    def _toggle_blank(self, e):
        self.state.engine.microscope.set_beam_blank(bool(e.value))
        status = "blanked" if e.value else "unblanked"
        ui.notify(f"Beam {status}.", type="info")

    def _jog_stage(self, dx: float, dy: float):
        self.state.engine.microscope.set_stage_relative(dx, dy)
        self.refresh_telemetry()

    def _run_autofocus(self):
        try:
            opt_olf, score = self.state.engine.autofocus.run_autofocus()
            if self.autofocus_res_label:
                self.autofocus_res_label.set_text(f"Optimized: OLf={opt_olf} (sharpness={score:.1f})")
            self.refresh_telemetry()
            ui.notify(f"Autofocus complete: OLf set to {opt_olf}.", type="positive")
        except Exception as e:
            ui.notify(f"Autofocus error: {e}", type="negative")

    def _load_plugin(self):
        path = self.plugin_path_input.value.strip()
        if not path:
            ui.notify("Please enter a valid Python script file path.", type="warning")
            return

        try:
            discovered = GLOBAL_REGISTRY.load_from_file(path)
            # Register discovered filters into the pipeline
            for filter_name in discovered.get("filter", []):
                filt_func = GLOBAL_REGISTRY.filter_routines[filter_name]
                self.state.engine.pipeline.register_filter(filter_name, filt_func)

            # Register discovered corrections into the engine
            for corr_name in discovered.get("correction", []):
                corr_cls = GLOBAL_REGISTRY.correction_routines[corr_name]
                if isinstance(corr_cls, type):
                    instance = corr_cls()
                    self.state.engine.corrections_manager.register(instance)
                elif isinstance(corr_cls, BaseCorrectionRoutine):
                    self.state.engine.corrections_manager.register(corr_cls)

            self._render_active_routines()
            self._render_engine_corrections()
            count = sum(len(v) for v in discovered.values())
            ui.notify(f"Loaded {count} routines from {Path(path).name}!", type="positive")
        except Exception as e:
            ui.notify(f"Failed to load plugin: {e}", type="negative")

    def _render_active_routines(self):
        if not hasattr(self, "routines_list"):
            return
        self.routines_list.clear()
        with self.routines_list:
            filt_names = list(GLOBAL_REGISTRY.filter_routines.keys())
            c_names = list(GLOBAL_REGISTRY.clustering_routines.keys())
            d_names = list(GLOBAL_REGISTRY.decision_routines.keys())
            f_names = list(GLOBAL_REGISTRY.live_feedback_routines.keys())
            corr_names = list(GLOBAL_REGISTRY.correction_routines.keys())

            ui.label(f"• Image / Threshold Filters: {', '.join(filt_names) if filt_names else 'None'}").classes("text-cyan-400")
            ui.label(f"• Clustering: {', '.join(c_names) if c_names else 'None'}")
            ui.label(f"• Decision: {', '.join(d_names) if d_names else 'None'}")
            ui.label(f"• Live Feedback: {', '.join(f_names) if f_names else 'None'}")
            ui.label(f"• Custom Corrections: {', '.join(corr_names) if corr_names else 'None'}")

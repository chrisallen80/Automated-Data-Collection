"""
Stage Navigator and Multi-Site Automated Collection Component.
Enables researchers to pre-select stage positions (by driving the microscope
and clicking 'Record Current Position' or entering coordinates),
then automatically iterate through every position running survey, thresholding,
clustering, and tile acquisition.
"""

from __future__ import annotations
from typing import Optional, Callable
from nicegui import ui

from ..state import UIState
from adc.acquisition.multi_site import StageSite, SiteStatus


class StageNavigator:
    def __init__(self, state: UIState, on_site_selected: Optional[Callable[[StageSite], None]] = None):
        self.state = state
        self.on_site_selected = on_site_selected
        self.table: Optional[ui.table] = None
        self.progress_bar: Optional[ui.linear_progress] = None
        self.progress_label: Optional[ui.label] = None
        self.curr_pos_label: Optional[ui.label] = None

    def build(self) -> ui.card:
        with ui.card().classes("w-full p-4 bg-slate-900 border border-slate-800 rounded-xl shadow-lg"):
            # Header
            with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-800"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("navigation", size="sm", color="accent")
                    ui.label("Stage Navigator & Multi-Site Batch Acquisition").classes("text-lg font-semibold text-slate-100")

                with ui.row().classes("gap-2 items-center"):
                    ui.button("Record Current Position", icon="add_location_alt", on_click=self._record_current_position).props("unelevated color=accent")
                    ui.button("Clear All", icon="delete_sweep", on_click=self._clear_all).props("flat color=negative dense")

            # Current Stage Telemetry & Jog
            with ui.row().classes("w-full items-center justify-between bg-slate-950/40 p-3 rounded-lg border border-slate-800/60 my-2"):
                with ui.column().classes("gap-0"):
                    ui.label("Current Stage Coordinates:").classes("text-xs text-slate-400")
                    pos = self.state.engine.microscope.get_stage_position()
                    self.curr_pos_label = ui.label(f"X: {pos.x_nm/1000.0:.3f} μm | Y: {pos.y_nm/1000.0:.3f} μm | Z: {pos.z_nm/1000.0:.3f} μm").classes("text-sm font-mono text-cyan-400")

                with ui.row().classes("gap-1 items-center"):
                    ui.button("← -10μm", on_click=lambda: self._jog(-10000, 0)).props("dense flat outline size=sm")
                    ui.button("→ +10μm", on_click=lambda: self._jog(10000, 0)).props("dense flat outline size=sm")
                    ui.button("↓ -10μm", on_click=lambda: self._jog(0, -10000)).props("dense flat outline size=sm")
                    ui.button("↑ +10μm", on_click=lambda: self._jog(0, 10000)).props("dense flat outline size=sm")

            # Waypoint Table
            columns = [
                {"name": "id", "label": "ID", "field": "id", "sortable": True, "align": "center"},
                {"name": "label", "label": "Label", "field": "label", "align": "left"},
                {"name": "x_um", "label": "X (μm)", "field": "x_um", "sortable": True, "align": "right"},
                {"name": "y_um", "label": "Y (μm)", "field": "y_um", "sortable": True, "align": "right"},
                {"name": "status", "label": "Status", "field": "status", "align": "center"},
                {"name": "clusters", "label": "Features", "field": "clusters", "align": "right"},
                {"name": "tiles", "label": "Tiles", "field": "tiles", "align": "right"},
                {"name": "action", "label": "Drive / Delete", "field": "action", "align": "center"},
            ]

            self.table = ui.table(
                columns=columns,
                rows=self._get_table_rows(),
                row_key="id",
                pagination={"rowsPerPage": 6}
            ).classes("w-full bg-slate-950/60 rounded border border-slate-800")

            # Slot for action buttons inside table
            self.table.add_slot(
                "body-cell-action",
                """
                <q-td :props="props">
                    <q-btn dense flat round color="accent" icon="my_location" @click="$parent.$emit('drive_to', props.row.raw_id)">
                        <q-tooltip>Drive stage to this position</q-tooltip>
                    </q-btn>
                    <q-btn dense flat round color="negative" icon="delete" @click="$parent.$emit('delete_site', props.row.raw_id)">
                        <q-tooltip>Remove waypoint</q-tooltip>
                    </q-btn>
                </q-td>
                """
            )
            self.table.on("drive_to", lambda e: self._drive_to_site(e.args))
            self.table.on("delete_site", lambda e: self._delete_site(e.args))

            # Batch Execution Controls & Progress
            with ui.column().classes("w-full bg-slate-950/60 p-4 rounded-lg border border-slate-800 mt-2 gap-3"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("gap-2 items-center"):
                        self.btn_run = ui.button("Run Multi-Site Batch", icon="play_circle_filled", on_click=self._run_multi_site).props("unelevated color=accent")
                        self.btn_stop = ui.button("Stop Multi-Site", icon="stop", on_click=self._stop_multi_site).props("flat color=negative")

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Multi-Site Progress:").classes("text-xs text-slate-400")
                        self.progress_label = ui.label("0/0 Sites Completed").classes("text-xs font-mono text-slate-200")

                self.progress_bar = ui.linear_progress(value=0.0, show_value=False).props("color=accent size=10px rounded")

        # Subscribe to multi-site callbacks
        self.state.engine.subscribe("on_multi_site_progress", self._on_multi_site_progress)
        self.state.engine.subscribe("on_multi_site_completed", self._on_multi_site_completed)

        return self

    def _get_table_rows(self):
        rows = []
        for s in self.state.engine.multi_site_manager.sites:
            rows.append({
                "raw_id": s.site_id,
                "id": f"#{s.site_id}",
                "label": s.label,
                "x_um": f"{s.x_nm / 1000.0:.3f}",
                "y_um": f"{s.y_nm / 1000.0:.3f}",
                "status": s.status.value,
                "clusters": s.clusters_found,
                "tiles": s.tiles_collected,
                "action": "",
            })
        return rows

    def update_table(self):
        if self.table:
            self.table.rows = self._get_table_rows()
            self.table.update()
        self._update_telemetry()

    def _update_telemetry(self):
        if self.curr_pos_label:
            pos = self.state.engine.microscope.get_stage_position()
            self.curr_pos_label.set_text(f"X: {pos.x_nm/1000.0:.3f} μm | Y: {pos.y_nm/1000.0:.3f} μm | Z: {pos.z_nm/1000.0:.3f} μm")

    def _record_current_position(self):
        pos = self.state.engine.microscope.get_stage_position()
        site = self.state.engine.multi_site_manager.add_from_position(pos)
        self.update_table()
        ui.notify(f"Recorded Site #{site.site_id} at ({pos.x_nm/1000.0:.2f}, {pos.y_nm/1000.0:.2f}) μm", type="positive")

    def _jog(self, dx_nm: float, dy_nm: float):
        self.state.engine.microscope.set_stage_relative(dx_nm, dy_nm)
        self._update_telemetry()

    def _drive_to_site(self, site_id: int):
        site = self.state.engine.multi_site_manager.get_site(site_id)
        if site:
            self.state.engine.microscope.set_stage_position(site.x_nm, site.y_nm)
            self._update_telemetry()
            ui.notify(f"Drove stage to {site.label} ({site.x_nm/1000.0:.2f}, {site.y_nm/1000.0:.2f}) μm", type="info")

    def _delete_site(self, site_id: int):
        self.state.engine.multi_site_manager.remove_site(site_id)
        self.update_table()
        ui.notify(f"Removed Site #{site_id}.", type="info")

    def _clear_all(self):
        self.state.engine.multi_site_manager.clear()
        self.update_table()
        ui.notify("Cleared all recorded stage waypoints.", type="warning")

    def _run_multi_site(self):
        try:
            if not self.state.engine.multi_site_manager.sites:
                ui.notify("Please record at least one stage position first.", type="warning")
                return

            self.state.engine.multi_site_manager.reset_statuses()
            self.update_table()
            self.state.engine.start_multi_site_acquisition()
            self.btn_run.disable()
            ui.notify("Multi-Site Batch Acquisition started!", type="positive")
        except Exception as e:
            ui.notify(f"Error starting multi-site batch: {e}", type="negative")

    def _stop_multi_site(self):
        self.state.engine.stop_acquisition()
        self.btn_run.enable()
        ui.notify("Multi-Site acquisition stopped.", type="negative")

    def _on_multi_site_progress(self, current: int, total: int, site: StageSite):
        frac = float(current) / float(total) if total > 0 else 0.0
        if self.progress_bar:
            self.progress_bar.set_value(frac)
        if self.progress_label:
            self.progress_label.set_text(f"Site {current}/{total}: {site.label} ({site.status.value})")
        self.update_table()

    def _on_multi_site_completed(self):
        if self.btn_run:
            self.btn_run.enable()
        if self.progress_label:
            self.progress_label.set_text("Batch Completed!")
        self.update_table()
        ui.notify("All preselected stage positions acquired successfully!", type="positive")

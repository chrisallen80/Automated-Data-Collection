"""
Feature Segmentation and Cluster Selection Panel.
Provides clustering controls, cluster table, and ROI selection.
"""

from __future__ import annotations
from typing import Callable, Optional
from nicegui import ui
from ..state import UIState


class ClusterPanel:
    def __init__(self, state: UIState, on_clusters_changed: Optional[Callable[[], None]] = None):
        self.state = state
        self.on_clusters_changed = on_clusters_changed
        self.table: Optional[ui.table] = None

    def build(self) -> ui.card:
        with ui.card().classes("w-full p-4 bg-slate-900 border border-slate-800 rounded-xl shadow-lg"):
            with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-800"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("bubble_chart", size="sm", color="positive")
                    ui.label("Feature Segmentation & Cluster Manager").classes("text-lg font-semibold text-slate-100")

                with ui.row().classes("gap-2"):
                    ui.button("Select All", icon="check_box", on_click=self._select_all).props("flat dense color=positive")
                    ui.button("Clear Selection", icon="check_box_outline_blank", on_click=self._clear_selection).props("flat dense color=slate-400")
                    ui.button("Segment Features", icon="auto_awesome", on_click=self._run_detection).props("unelevated color=positive")

            # Algorithm parameter controls
            with ui.row().classes("w-full gap-4 items-center bg-slate-950/40 p-3 rounded-lg border border-slate-800/60 my-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("Method:").classes("text-xs text-slate-400")
                    self.method_select = ui.select(
                        ["contours", "dbscan"], value="contours", on_change=self._on_method_change
                    ).props("dense outlined options-dense")

                with ui.row().classes("items-center gap-2"):
                    ui.label("Min Area (px):").classes("text-xs text-slate-400")
                    self.min_area_input = ui.number(value=20, min=1, step=10).props("dense outlined").classes("w-24")

                with ui.row().classes("items-center gap-2"):
                    ui.label("Max Area (px):").classes("text-xs text-slate-400")
                    self.max_area_input = ui.number(value=200000, min=100, step=1000).props("dense outlined").classes("w-28")

                self.dbscan_params_row = ui.row().classes("items-center gap-2")
                with self.dbscan_params_row:
                    ui.label("DBSCAN eps:").classes("text-xs text-slate-400")
                    self.eps_input = ui.number(value=3.5, min=0.5, step=0.5).props("dense outlined").classes("w-20")

                self.dbscan_params_row.set_visibility(False)

            # Cluster list table
            columns = [
                {"name": "selected", "label": "Target", "field": "selected", "align": "center"},
                {"name": "id", "label": "ID", "field": "id", "sortable": True, "align": "center"},
                {"name": "center", "label": "Center (X, Y)", "field": "center", "align": "center"},
                {"name": "bbox", "label": "BBox (W x H)", "field": "bbox", "align": "center"},
                {"name": "area", "label": "Area (px)", "field": "area", "sortable": True, "align": "right"},
                {"name": "intensity", "label": "Mean Contrast", "field": "intensity", "sortable": True, "align": "right"},
            ]

            self.table = ui.table(
                columns=columns,
                rows=self._get_table_rows(),
                row_key="id",
                pagination={"rowsPerPage": 8}
            ).classes("w-full bg-slate-950/60 rounded border border-slate-800")

        return self

    def _on_method_change(self, e):
        is_dbscan = (e.value == "dbscan")
        self.dbscan_params_row.set_visibility(is_dbscan)

    def _get_table_rows(self):
        rows = []
        for c in self.state.engine.detected_clusters:
            is_sel = c.id in self.state.selected_cluster_ids if self.state.selected_cluster_ids else c.selected
            rows.append({
                "selected": "✓" if is_sel else "",
                "id": f"#{c.id}",
                "center": f"{c.center[0]:.1f}, {c.center[1]:.1f}",
                "bbox": f"{c.bbox[2]} x {c.bbox[3]}",
                "area": c.area_px,
                "intensity": f"{c.mean_intensity:.1f}",
            })
        return rows

    def _select_all(self):
        for c in self.state.engine.detected_clusters:
            c.selected = True
        self.state.selected_cluster_ids = {c.id for c in self.state.engine.detected_clusters}
        self.update_table()
        if self.on_clusters_changed:
            self.on_clusters_changed()

    def _clear_selection(self):
        for c in self.state.engine.detected_clusters:
            c.selected = False
        self.state.selected_cluster_ids.clear()
        self.update_table()
        if self.on_clusters_changed:
            self.on_clusters_changed()

    def _run_detection(self):
        if self.state.engine.raw_survey_image is None:
            ui.notify("Please acquire or load a survey image first.", type="warning")
            return

        method = self.method_select.value
        self.state.engine.clusterer.min_area = int(self.min_area_input.value)
        self.state.engine.clusterer.max_area = int(self.max_area_input.value)
        if method == "dbscan":
            self.state.engine.clusterer.eps = float(self.eps_input.value)

        clusters = self.state.engine.detect_features(method=method)
        self.state.selected_cluster_ids = {c.id for c in clusters}
        self.update_table()

        if self.on_clusters_changed:
            self.on_clusters_changed()

        ui.notify(f"Successfully segmented {len(clusters)} features via {method}.", type="positive")

    def update_table(self):
        if self.table:
            self.table.rows = self._get_table_rows()
            self.table.update()

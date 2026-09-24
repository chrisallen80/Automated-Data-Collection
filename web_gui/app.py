"""
Modern Web GUI Application for Automated Data Collection (ADC).
Built with NiceGUI, responsive Quasar components, and Tailwind CSS.
"""

from __future__ import annotations
import sys
from pathlib import Path
from nicegui import ui, app

from adc.acquisition.engine import AcquisitionEngine, AcquisitionState
from .state import UIState
from .components import (
    CanvasView,
    PipelinePanel,
    ClusterPanel,
    ScanPanel,
    LiveSTEM4DView,
    CorrectionsView,
    StageNavigator,
)


def create_adc_app(engine: AcquisitionEngine, title: str = "Automated Data Collection (ADC)"):
    """
    Assemble the full browser application interface.
    """
    state = UIState(engine)

    # Dark mode electron microscopy theme
    ui.dark_mode().enable()

    # Apply global styles
    ui.add_head_html("""
    <style>
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background-color: #0b0f19; }
        .q-tab--active { color: #38bdf8 !important; font-weight: 600; }
    </style>
    """)

    # Top Navigation Bar
    with ui.header().classes("bg-slate-950 border-b border-slate-800 px-6 py-2 items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("biotech", size="md", color="primary")
            with ui.column().classes("gap-0"):
                ui.label(title).classes("text-lg font-bold text-slate-100 tracking-wide")
                ui.label("Automated 4D-STEM & Feature-Targeted Acquisition").classes("text-xs text-slate-400")

        with ui.row().classes("items-center gap-4"):
            # Hardware badge
            is_mock = engine.microscope.get_all_parameters().get("mock_mode", False)
            badge_color = "warning" if is_mock else "positive"
            badge_text = "SIMULATOR (MOCK)" if is_mock else "ONLINE (PyJEM / ARM200F)"
            with ui.row().classes("items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono"):
                ui.icon("radio_button_checked", size="xs", color=badge_color)
                ui.label(badge_text).classes(f"text-{badge_color}")

            # Global Survey trigger
            survey_size_select = ui.select([256, 512, 1024, 2048], value=512, label="Survey Size (px)").props("dense outlined options-dense borderless")
            ui.button("New Survey STEM", icon="radar", on_click=lambda: _acquire_survey_and_refresh()).props("unelevated color=primary no-caps")

    # Main Tab Layout
    with ui.column().classes("w-full max-w-7xl mx-auto p-4 gap-4"):
        with ui.tabs().classes("w-full text-slate-400 border-b border-slate-800") as tabs:
            tab_stage = ui.tab("Stage Navigator", icon="navigation")
            tab_survey = ui.tab("Survey & Pipeline", icon="tune")
            tab_clusters = ui.tab("Feature Clusters", icon="bubble_chart")
            tab_scan = ui.tab("Scan Grid & Queue", icon="grid_4x4")
            tab_stem4d = ui.tab("4D-STEM & Merlin", icon="lens")
            tab_hw = ui.tab("Microscope & Plugins", icon="precision_manufacturing")

        # Global shared Canvas View instance
        canvas = CanvasView(state)

        # Tab Panels
        with ui.tab_panels(tabs, value=tab_stage).classes("w-full bg-transparent p-0"):

            # Tab 0: Stage Navigator & Multi-Site Batch
            with ui.tab_panel(tab_stage).classes("p-0"):
                with ui.grid(columns=2).classes("w-full gap-4"):
                    canvas_stage = CanvasView(state).build()
                    stage_navigator = StageNavigator(state).build()

            # Tab 1: Survey & Preprocessing Pipeline
            with ui.tab_panel(tab_survey).classes("p-0"):
                with ui.grid(columns=2).classes("w-full gap-4"):
                    canvas.build()
                    pipeline_panel = PipelinePanel(state, on_pipeline_updated=canvas.update).build()

            # Tab 2: Feature Segmentation & Clustering
            with ui.tab_panel(tab_clusters).classes("p-0"):
                with ui.grid(columns=2).classes("w-full gap-4"):
                    canvas_copy = CanvasView(state).build()
                    cluster_panel = ClusterPanel(state, on_clusters_changed=lambda: (canvas.update(), canvas_copy.update())).build()

            # Tab 3: Scan Grid Planning & Execution
            with ui.tab_panel(tab_scan).classes("p-0"):
                with ui.grid(columns=2).classes("w-full gap-4"):
                    canvas_copy2 = CanvasView(state).build()
                    scan_panel = ScanPanel(state, on_plan_updated=lambda: (canvas.update(), canvas_copy2.update())).build()

            # Tab 4: 4D-STEM Diffraction & Virtual Detectors
            with ui.tab_panel(tab_stem4d).classes("p-0"):
                stem4d_view = LiveSTEM4DView(state).build()

            # Tab 5: Hardware Telemetry, Drift & Plugins
            with ui.tab_panel(tab_hw).classes("p-0"):
                corrections_view = CorrectionsView(state).build()

        # Bottom Rolling Activity Log
        with ui.expansion("Experiment Activity Log", icon="terminal").classes("w-full bg-slate-900 border border-slate-800 rounded-xl shadow mt-2"):
            log_container = ui.log(max_lines=100).classes("w-full h-32 bg-slate-950 font-mono text-xs text-slate-300 p-2 rounded")

    def _on_engine_log(text: str):
        log_container.push(text)

    engine.subscribe("on_log", _on_engine_log)

    def _acquire_survey_and_refresh():
        try:
            sz = survey_size_select.value
            engine.acquire_survey(size=(sz, sz))
            engine.process_survey()
            canvas.update()
            ui.notify(f"New {sz}x{sz} survey micrograph acquired.", type="positive")
        except Exception as e:
            ui.notify(f"Survey acquisition error: {e}", type="negative")

    # Acquire initial survey if starting in mock mode
    if is_mock and engine.raw_survey_image is None:
        try:
            sz = survey_size_select.value
            engine.acquire_survey(size=(sz, sz))
            engine.process_survey()
            canvas.update()
        except Exception:
            pass


def start_server(engine: AcquisitionEngine, host: str = "127.0.0.1", port: int = 8080, title: str = "Automated Data Collection"):
    """Launch NiceGUI web server."""
    create_adc_app(engine, title=title)
    ui.run(host=host, port=port, title=title, reload=False, show=False)

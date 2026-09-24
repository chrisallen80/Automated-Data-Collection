"""
Unit tests for In-Flight Closed-Loop Corrections Framework:
- Independent drift and autofocus selection
- Configurable scan intervals (frequencies)
- Reference location modes (Previous Scan, Next Scan, Fixed Region)
- Pluggable custom correction routines
"""

import numpy as np
from adc.hardware.mock import MockMicroscope, MockScanner
from adc.imaging.grid_planner import ScanTile
from adc.imaging.clustering import Cluster
from adc.corrections import (
    CorrectionsManager,
    CorrectionLocationMode,
    CorrectionContext,
    CorrectionResult,
    BaseCorrectionRoutine,
    DriftTracker,
    DriftCorrectionRoutine,
    AutofocusController,
    AutofocusCorrectionRoutine,
)


def _make_dummy_context(scan_index: int, total_scans: int = 20, location_mode=CorrectionLocationMode.FIXED_REGION):
    scope = MockMicroscope()
    scope.set_beam_blank(False)
    scanner = MockScanner(microscope=scope)
    survey_img = scanner.acquire_survey_image(size=(128, 128))

    current_tile = ScanTile(
        tile_id=scan_index,
        cluster_id=1,
        x=32, y=32,
        width=32, height=32,
        row=0, col=0,
        stage_x_nm=1000.0, stage_y_nm=2000.0
    )
    prev_tile = ScanTile(
        tile_id=scan_index - 1,
        cluster_id=1,
        x=0, y=0,
        width=32, height=32,
        row=0, col=0,
        stage_x_nm=500.0, stage_y_nm=1500.0
    ) if scan_index > 0 else None

    cluster = Cluster(
        id=1,
        center=(64, 64),
        area_px=100,
        bbox=(48, 48, 32, 32),
        contour=np.array([[48, 48], [80, 80]])
    )

    return CorrectionContext(
        scan_index=scan_index,
        total_scans=total_scans,
        current_tile=current_tile,
        previous_tile=prev_tile,
        fixed_cluster=cluster,
        microscope=scope,
        scanner=scanner,
        survey_image=survey_img,
        nm_per_pixel=1.0
    )


def test_independent_selection_and_intervals():
    mgr = CorrectionsManager()

    # Drift configured to run every 3 scans
    drift = DriftCorrectionRoutine(enabled=True, interval_scans=3)
    # Autofocus configured to run every 5 scans, disabled initially
    focus = AutofocusCorrectionRoutine(enabled=False, interval_scans=5)

    mgr.register(drift)
    mgr.register(focus)

    # Scan 0: should not trigger (acquisition just starting)
    ctx0 = _make_dummy_context(scan_index=0)
    res0 = mgr.run_corrections_if_due(ctx0)
    assert len(res0) == 0

    # Scan 1: neither is due
    ctx1 = _make_dummy_context(scan_index=1)
    res1 = mgr.run_corrections_if_due(ctx1)
    assert len(res1) == 0

    # Scan 3: drift is due (3 % 3 == 0), focus is disabled
    ctx3 = _make_dummy_context(scan_index=3)
    res3 = mgr.run_corrections_if_due(ctx3)
    assert len(res3) == 1
    assert res3[0].routine_name == "drift"
    assert res3[0].executed is True

    # Now enable only focus and disable drift
    drift.enabled = False
    focus.enabled = True
    focus.interval_scans = 4

    # Scan 4: only focus is due (4 % 4 == 0)
    ctx4 = _make_dummy_context(scan_index=4)
    res4 = mgr.run_corrections_if_due(ctx4)
    assert len(res4) == 1
    assert res4[0].routine_name == "focus"
    assert res4[0].executed is True

    # Now enable BOTH: drift every 2 scans, focus every 4 scans
    drift.enabled = True
    drift.interval_scans = 2
    focus.enabled = True
    focus.interval_scans = 4

    # Scan 4: BOTH drift (4 % 2 == 0) and focus (4 % 4 == 0) are due
    ctx4_both = _make_dummy_context(scan_index=4)
    res4_both = mgr.run_corrections_if_due(ctx4_both)
    assert len(res4_both) == 2
    routine_names = [r.routine_name for r in res4_both]
    assert "drift" in routine_names
    assert "focus" in routine_names


def test_reference_location_modes():
    scope = MockMicroscope()
    scope.set_beam_blank(False)
    scanner = MockScanner(microscope=scope)
    survey_img = scanner.acquire_survey_image(size=(128, 128))

    cluster = Cluster(
        id=42,
        center=(50, 50),
        area_px=100,
        bbox=(35, 35, 30, 30),
        contour=np.array([[35, 35], [65, 65]])
    )

    tile = ScanTile(tile_id=1, cluster_id=42, x=20, y=20, width=20, height=20, row=0, col=0, stage_x_nm=5000.0, stage_y_nm=5000.0)
    prev = ScanTile(tile_id=0, cluster_id=42, x=0, y=0, width=20, height=20, row=0, col=0, stage_x_nm=1000.0, stage_y_nm=1000.0)

    # 1. FIXED_REGION mode
    drift_fixed = DriftCorrectionRoutine(
        enabled=True,
        interval_scans=1,
        location_mode=CorrectionLocationMode.FIXED_REGION,
        fixed_cluster_id=42
    )
    ctx_fixed = CorrectionContext(
        scan_index=1,
        total_scans=10,
        current_tile=tile,
        previous_tile=prev,
        fixed_cluster=cluster,
        microscope=scope,
        scanner=scanner,
        survey_image=survey_img,
        nm_per_pixel=1.0
    )
    res_fixed = drift_fixed.execute(ctx_fixed)
    assert res_fixed.executed is True
    assert "FIXED_REGION" in res_fixed.message

    # 2. PREVIOUS_SCAN mode
    drift_prev = DriftCorrectionRoutine(
        enabled=True,
        interval_scans=1,
        location_mode=CorrectionLocationMode.PREVIOUS_SCAN
    )
    res_prev = drift_prev.execute(ctx_fixed)
    assert res_prev.executed is True
    assert "PREVIOUS_SCAN" in res_prev.message

    # 3. NEXT_SCAN mode
    drift_next = DriftCorrectionRoutine(
        enabled=True,
        interval_scans=1,
        location_mode=CorrectionLocationMode.NEXT_SCAN
    )
    res_next = drift_next.execute(ctx_fixed)
    assert res_next.executed is True
    assert "NEXT_SCAN" in res_next.message

    # 4. ARBITRARY_POSITION mode (Stage coordinates in nm)
    drift_arb_nm = DriftCorrectionRoutine(
        enabled=True,
        interval_scans=1,
        location_mode=CorrectionLocationMode.ARBITRARY_POSITION,
        arbitrary_position_nm=(2500.0, 3500.0)
    )
    res_arb_nm = drift_arb_nm.execute(ctx_fixed)
    assert res_arb_nm.executed is True
    assert "ARBITRARY_POSITION" in res_arb_nm.message

    # 5. ARBITRARY_POSITION mode (Survey pixel coordinates)
    drift_arb_px = DriftCorrectionRoutine(
        enabled=True,
        interval_scans=1,
        location_mode=CorrectionLocationMode.ARBITRARY_POSITION,
        arbitrary_position_px=(75, 85)
    )
    res_arb_px = drift_arb_px.execute(ctx_fixed)
    assert res_arb_px.executed is True
    assert "ARBITRARY_POSITION" in res_arb_px.message

    # 6. Autofocus with ARBITRARY_POSITION
    focus_arb = AutofocusCorrectionRoutine(
        enabled=True,
        interval_scans=1,
        location_mode=CorrectionLocationMode.ARBITRARY_POSITION,
        arbitrary_position_nm=(1200.0, 1800.0)
    )
    res_focus_arb = focus_arb.execute(ctx_fixed)
    assert res_focus_arb.executed is True
    assert "ARBITRARY_POSITION" in res_focus_arb.message


def test_pluggable_custom_correction_routine():
    class CustomBeamTiltCorrection(BaseCorrectionRoutine):
        def __init__(self, interval_scans: int = 7):
            super().__init__(
                name="beam_tilt",
                enabled=True,
                interval_scans=interval_scans,
                location_mode=CorrectionLocationMode.NEXT_SCAN
            )
            self.execution_count = 0

        def execute(self, context: CorrectionContext) -> CorrectionResult:
            self.execution_count += 1
            return CorrectionResult(
                routine_name=self.name,
                executed=True,
                message=f"Custom beam tilt correction applied! Count={self.execution_count}",
                extra={"tilt_mrad": 0.05}
            )

    mgr = CorrectionsManager()
    custom_corr = CustomBeamTiltCorrection(interval_scans=7)
    mgr.register(custom_corr)

    # Run at scan 7
    ctx7 = _make_dummy_context(scan_index=7)
    results = mgr.run_corrections_if_due(ctx7)
    assert len(results) == 1
    assert results[0].routine_name == "beam_tilt"
    assert results[0].executed is True
    assert results[0].extra["tilt_mrad"] == 0.05
    assert custom_corr.execution_count == 1

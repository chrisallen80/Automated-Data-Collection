"""
Unit tests for Stage Navigator and Multi-Site Batch Acquisition.
"""

import tempfile
from pathlib import Path

from adc.hardware.mock import MockMicroscope, MockScanner, MockMerlin
from adc.hardware.base import StagePosition
from adc.acquisition.multi_site import MultiSiteManager, StageSite, SiteStatus
from adc.acquisition.engine import AcquisitionEngine, AcquisitionState


def test_multi_site_manager():
    mgr = MultiSiteManager()
    assert len(mgr.sites) == 0

    # Add manual site
    s1 = mgr.add_site(x_nm=10000.0, y_nm=20000.0, label="Grid_A1")
    assert s1.site_id == 1
    assert s1.label == "Grid_A1"
    assert s1.x_nm == 10000.0
    assert s1.status == SiteStatus.PENDING

    # Add site from StagePosition
    pos = StagePosition(x_nm=30000.0, y_nm=40000.0, z_nm=500.0)
    s2 = mgr.add_from_position(pos, label="Grid_A2")
    assert s2.site_id == 2
    assert s2.label == "Grid_A2"
    assert len(mgr.sites) == 2

    # Test JSON serialization & loading
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "waypoints.json"
        mgr.save_to_json(json_file)
        assert json_file.exists()

        mgr2 = MultiSiteManager()
        loaded_count = mgr2.load_from_json(json_file)
        assert loaded_count == 2
        assert mgr2.sites[0].label == "Grid_A1"
        assert mgr2.sites[1].x_nm == 30000.0

    # Test removal
    assert mgr.remove_site(1) is True
    assert len(mgr.sites) == 1
    assert mgr.get_site(1) is None
    assert mgr.get_site(2) is not None


def test_multi_site_batch_execution():
    scope = MockMicroscope()
    scanner = MockScanner(microscope=scope)
    camera = MockMerlin(detector_size=128)

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = AcquisitionEngine(
            microscope=scope,
            scanner=scanner,
            camera=camera,
            save_dir=tmpdir
        )

        # Preselect 2 stage positions
        site1 = engine.multi_site_manager.add_site(x_nm=5000.0, y_nm=15000.0, label="Site_1")
        site2 = engine.multi_site_manager.add_site(x_nm=25000.0, y_nm=35000.0, label="Site_2")

        progress_events = []
        def on_prog(curr, total, site):
            progress_events.append((curr, total, site.site_id, site.status))

        engine.subscribe("on_multi_site_progress", on_prog)

        # Run multi-site loop synchronously for testing
        engine._run_multi_site_loop(settling_time_s=0.01)

        # Verify both sites were visited and completed
        assert site1.status == SiteStatus.COMPLETED
        assert site2.status == SiteStatus.COMPLETED
        assert site1.tiles_collected > 0
        assert site2.tiles_collected > 0

        # Verify stage ended at Site 2 coordinates (within survey FOV)
        final_pos = scope.get_stage_position()
        assert abs(final_pos.x_nm - 25000.0) < 1000.0
        assert abs(final_pos.y_nm - 35000.0) < 1000.0

        # Verify progress callbacks fired
        assert len(progress_events) >= 2
        assert engine.state == AcquisitionState.COMPLETED

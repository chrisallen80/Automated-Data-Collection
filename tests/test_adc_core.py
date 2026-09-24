"""
Comprehensive unit test suite for ADC core engine.
"""

import tempfile
from pathlib import Path
import numpy as np

from adc.hardware.mock import MockMicroscope, MockScanner, MockMerlin, MockEDX
from adc.hardware.base import StagePosition
from adc.imaging import ImagePipeline, FeatureClusterer, GridPlanner, STEM4DProcessor
from adc.scanning import PatternGenerator
from adc.corrections import DriftTracker, AutofocusController, EmissionMonitor
from adc.plugins import GLOBAL_REGISTRY, PluginRegistry, adc_clustering, ScanTarget
from adc.acquisition import AcquisitionEngine, AcquisitionState, HDF5Writer


def test_mock_hardware():
    scope = MockMicroscope()
    assert scope.is_connected()
    pos = scope.get_stage_position()
    assert isinstance(pos, StagePosition)
    
    # Test stage movement
    new_pos = scope.set_stage_position(12000.0, 18000.0)
    assert abs(new_pos.x_nm - 12000.0) < 1.0
    assert abs(new_pos.y_nm - 18000.0) < 1.0

    # Test defocus & beam blank
    scope.set_defocus(39000)
    assert scope.get_defocus() == 39000
    scope.set_beam_blank(True)
    assert scope.get_beam_blank() is True

    # Test scanner
    scanner = MockScanner(microscope=scope)
    # When blanked, image should be dark
    dark_img = scanner.acquire_survey_image(size=(128, 128))
    assert dark_img.shape == (128, 128)
    assert np.mean(dark_img) < 20.0

    # Unblank and acquire realistic frame
    scope.set_beam_blank(False)
    img = scanner.acquire_survey_image(size=(128, 128))
    assert img.shape == (128, 128)
    assert np.mean(img) > 20.0

    # Test Merlin camera
    camera = MockMerlin(detector_size=128)
    assert camera.is_connected()
    frame = camera.get_latest_frame()
    assert frame is not None
    assert frame.shape == (128, 128)

    # Test EDX
    edx = MockEDX(num_channels=512)
    energies, counts = edx.acquire_point_spectrum(dwell_s=0.5)
    assert len(energies) == 512
    assert len(counts) == 512


def test_image_pipeline_and_clustering():
    pipeline = ImagePipeline()
    # Create synthetic test image with two distinct bright circular discs
    img = np.zeros((256, 256), dtype=np.uint8)
    # Disc 1 at (60, 60), radius 20
    y, x = np.ogrid[:256, :256]
    mask1 = ((x - 60) ** 2 + (y - 60) ** 2) <= 20 ** 2
    # Disc 2 at (180, 180), radius 25
    mask2 = ((x - 180) ** 2 + (y - 180) ** 2) <= 25 ** 2
    img[mask1] = 220
    img[mask2] = 240

    processed = pipeline.process(img)
    assert processed.shape == (256, 256)
    assert processed.dtype == np.uint8

    # Cluster detection
    clusterer = FeatureClusterer(min_area=50, max_area=10000)
    clusters = clusterer.segment_contours(processed, original_img=img)
    assert len(clusters) == 2

    # Check bounding box centroids
    c_centers = [c.center for c in clusters]
    assert any(abs(c[0] - 60) < 5 and abs(c[1] - 60) < 5 for c in c_centers)
    assert any(abs(c[0] - 180) < 5 and abs(c[1] - 180) < 5 for c in c_centers)


def test_grid_planner():
    clusterer = FeatureClusterer()
    # Mock cluster of size 300x150
    from adc.imaging.clustering import Cluster
    c = Cluster(id=0, bbox=(50, 50, 300, 150), center=(200, 125), area_px=45000)
    
    planner = GridPlanner(tile_size=128, overlap_ratio=0.10)
    tiles = planner.plan_cluster_tiles(c, survey_shape=(512, 512), tighten=False)
    assert len(tiles) >= 3
    # Check all tiles are within boundaries
    for t in tiles:
        assert 0 <= t.x <= 512 - 128
        assert 0 <= t.y <= 512 - 128
        assert t.width == 128
        assert t.height == 128


def test_stem4d_processing():
    processor = STEM4DProcessor()
    # Create synthetic diffraction pattern
    pattern = np.zeros((128, 128), dtype=np.float32)
    # Center disc
    y, x = np.ogrid[:128, :128]
    r = np.sqrt((x - 64) ** 2 + (y - 64) ** 2)
    pattern[r <= 15] = 1000.0
    # Add Bragg spots in annulus (r=35)
    pattern[((x - 64 - 35) ** 2 + (y - 64) ** 2) <= 9] = 2000.0
    pattern[((x - 64 + 35) ** 2 + (y - 64) ** 2) <= 9] = 2000.0

    # VBF integration
    vbf = processor.integrate_virtual_aperture(pattern, inner_radius=0, outer_radius=15, center=(64, 64))
    assert vbf > 100000.0

    # VADF integration
    vadf = processor.integrate_virtual_aperture(pattern, inner_radius=25, outer_radius=50, center=(64, 64))
    assert vadf > 50000.0

    # Center of mass
    com_x, com_y = processor.compute_center_of_mass(pattern, center=(64, 64))
    assert abs(com_x) < 2.0  # symmetric spots
    assert abs(com_y) < 2.0

    # Crystallinity index
    cryst = processor.compute_crystallinity_index(pattern, central_disk_radius=15, outer_annulus_radius=50, center=(64, 64))
    assert cryst["is_crystalline"] is True


def test_drift_tracker():
    tracker = DriftTracker(ref_bbox=(50, 50, 64, 64), upsample_factor=10, nm_per_pixel=1.0)
    
    # Base image with texture
    rng = np.random.default_rng(123)
    base = rng.normal(128, 30, size=(256, 256)).astype(np.uint8)
    tracker.set_reference(base)

    # Shifted image by dx=+4, dy=+3
    shifted = np.roll(np.roll(base, 3, axis=0), 4, axis=1)
    dx_nm, dy_nm = tracker.measure_drift(shifted)
    
    # Check measured drift within 0.3 pixel/nm
    assert abs(dx_nm - 4.0) < 0.5
    assert abs(dy_nm - 3.0) < 0.5


def test_autofocus():
    scope = MockMicroscope(initial_olf=38942)
    scanner = MockScanner(microscope=scope)
    autofocus = AutofocusController(scope, scanner, sweep_range=30, sweep_steps=7)

    # Put microscope off-focus
    scope.set_defocus(38920)
    best_olf, score = autofocus.run_autofocus(target_size=(128, 128))
    # Optimal focus should be close to 38942
    assert abs(best_olf - 38942) <= 15
    assert scope.get_defocus() == best_olf


def test_plugin_dynamic_loading():
    registry = PluginRegistry()
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin_file = Path(tmpdir) / "custom_routine.py"
        plugin_file.write_text(
            'from adc.plugins import adc_clustering\n'
            'from adc.imaging.clustering import Cluster\n'
            '@adc_clustering\n'
            'def cluster_my_custom_detector(image, config):\n'
            '    return [Cluster(id=99, bbox=(10, 10, 50, 50), center=(35, 35), area_px=2500)]\n'
        )
        discovered = registry.load_from_file(plugin_file)
        assert "cluster_my_custom_detector" in discovered["clustering"]
        assert "cluster_my_custom_detector" in registry.clustering_routines


def test_full_acquisition_engine_workflow():
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

        # 1. Acquire survey
        survey = engine.acquire_survey(size=(256, 256))
        assert survey.shape == (256, 256)

        # 2. Process pipeline & detect features
        clusters = engine.detect_features(method="contours")
        assert len(clusters) > 0

        # 3. Plan scan tiles
        tiles = engine.plan_scan_tiles(tile_size=128, overlap=0.10)
        assert len(tiles) > 0

        # 4. Limit to first 2 tiles for rapid test
        engine.planned_tiles = tiles[:2]

        # 5. Execute acquisition synchronously
        engine._run_acquisition_loop()

        assert engine.state == AcquisitionState.COMPLETED
        assert all(t.completed for t in engine.planned_tiles)

        # 6. Verify HDF5 file created
        h5_files = list(Path(tmpdir).glob("*.h5"))
        assert len(h5_files) == 1

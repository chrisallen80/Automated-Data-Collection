"""
High-Level Acquisition Engine and State Machine.
Coordinates hardware, imaging pipeline, tile queue, drift correction, autofocus,
and HDF5 archival.
"""

from __future__ import annotations
import enum
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np

from ..hardware.base import BaseMicroscope, BaseScanner, BaseCamera, BaseEDX, StagePosition
from ..imaging.pipeline import ImagePipeline
from ..imaging.clustering import FeatureClusterer, Cluster
from ..imaging.grid_planner import GridPlanner, ScanTile
from ..corrections import (
    DriftTracker,
    AutofocusController,
    EmissionMonitor,
    CorrectionsManager,
    CorrectionLocationMode,
    CorrectionContext,
    CorrectionResult,
    DriftCorrectionRoutine,
    AutofocusCorrectionRoutine,
    BaseCorrectionRoutine
)
from ..plugins.base import ScanTarget, DecisionAction, LiveFeedbackFunc
from .live_feedback import LiveFeedbackManager
from .hdf5_writer import HDF5Writer
from .multi_site import MultiSiteManager, StageSite, SiteStatus


class AcquisitionState(str, enum.Enum):
    IDLE = "IDLE"
    SURVEYING = "SURVEYING"
    ANALYZING = "ANALYZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class AcquisitionEngine:
    """
    Central orchestrator for automated microscopy experiments.
    Decoupled from any graphical toolkit; works equally well in headless scripts
    or web UI servers.
    """

    def __init__(
        self,
        microscope: BaseMicroscope,
        scanner: BaseScanner,
        camera: Optional[BaseCamera] = None,
        edx: Optional[BaseEDX] = None,
        save_dir: str = "./acquisitions"
    ):
        self.microscope = microscope
        self.scanner = scanner
        self.camera = camera
        self.edx = edx
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # State management
        self.state: AcquisitionState = AcquisitionState.IDLE
        self.status_message: str = "Ready"
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._pause_requested = threading.Event()

        # Imaging and planning components
        self.pipeline = ImagePipeline()
        self.clusterer = FeatureClusterer()
        self.planner = GridPlanner(tile_size=256, overlap_ratio=0.10)

        # In-Flight Corrections Framework
        self.drift_tracker = DriftTracker()
        self.autofocus = AutofocusController(microscope, scanner)
        self.corrections_manager = CorrectionsManager()
        self.drift_routine = DriftCorrectionRoutine(
            tracker=self.drift_tracker,
            enabled=True,
            interval_scans=3,
            location_mode=CorrectionLocationMode.FIXED_REGION
        )
        self.autofocus_routine = AutofocusCorrectionRoutine(
            controller=self.autofocus,
            enabled=False,
            interval_scans=10,
            location_mode=CorrectionLocationMode.NEXT_SCAN
        )
        self.corrections_manager.register(self.drift_routine)
        self.corrections_manager.register(self.autofocus_routine)

        self.emission_monitor = EmissionMonitor(microscope)
        self.feedback_manager = LiveFeedbackManager()
        self.multi_site_manager = MultiSiteManager()

        # Runtime data caches
        self.raw_survey_image: Optional[np.ndarray] = None
        self.processed_survey_image: Optional[np.ndarray] = None
        self.detected_clusters: List[Cluster] = []
        self.planned_tiles: List[ScanTile] = []
        self.current_tile_index: int = 0
        self.latest_diffraction_frame: Optional[np.ndarray] = None
        self.arbitrary_correction_position_nm: Optional[Tuple[float, float]] = None
        self.arbitrary_correction_position_px: Optional[Tuple[int, int]] = None

        # Event callbacks
        self.callbacks: Dict[str, List[Callable[..., Any]]] = {
            "on_state_change": [],
            "on_survey_acquired": [],
            "on_clusters_updated": [],
            "on_tiles_updated": [],
            "on_tile_progress": [],
            "on_frame_received": [],
            "on_multi_site_progress": [],
            "on_multi_site_completed": [],
            "on_log": []
        }

    # Backward-compatible properties for corrections
    @property
    def enable_drift_correction(self) -> bool:
        return self.drift_routine.enabled

    @enable_drift_correction.setter
    def enable_drift_correction(self, val: bool) -> None:
        self.drift_routine.enabled = bool(val)

    @property
    def drift_check_interval(self) -> int:
        return self.drift_routine.interval_scans

    @drift_check_interval.setter
    def drift_check_interval(self, val: int) -> None:
        self.drift_routine.interval_scans = max(1, int(val))

    @property
    def enable_autofocus(self) -> bool:
        return self.autofocus_routine.enabled

    @enable_autofocus.setter
    def enable_autofocus(self, val: bool) -> None:
        self.autofocus_routine.enabled = bool(val)

    @property
    def autofocus_interval(self) -> int:
        return self.autofocus_routine.interval_scans

    @autofocus_interval.setter
    def autofocus_interval(self, val: int) -> None:
        self.autofocus_routine.interval_scans = max(1, int(val))

    def subscribe(self, event_name: str, callback: Callable[..., Any]) -> None:
        """Register a callback for engine events."""
        if event_name in self.callbacks:
            self.callbacks[event_name].append(callback)

    def _emit(self, event_name: str, *args, **kwargs) -> None:
        for cb in self.callbacks.get(event_name, []):
            try:
                cb(*args, **kwargs)
            except Exception as e:
                print(f"[AcquisitionEngine Callback Error] {event_name}: {e}")

    def _set_state(self, new_state: AcquisitionState, message: str = "") -> None:
        with self._lock:
            self.state = new_state
            if message:
                self.status_message = message
        self._emit("on_state_change", new_state, self.status_message)
        self._log(f"State -> {new_state.value}: {message}")

    def _log(self, text: str) -> None:
        self._emit("on_log", text)

    # ---------------- Survey & Pipeline ----------------

    def acquire_survey(self, size: Tuple[int, int] = (512, 512), dwell_s: float = 15e-6) -> np.ndarray:
        """Acquire a 2D survey image and initialize drift reference."""
        self._set_state(AcquisitionState.SURVEYING, "Acquiring survey image...")
        try:
            self.microscope.set_beam_blank(False)
            img = self.scanner.acquire_survey_image(size=size, dwell_time_s=dwell_s)
            self.raw_survey_image = img
            # Initialize drift reference
            self.drift_tracker.set_reference(img)
            self._set_state(AcquisitionState.IDLE, "Survey acquired successfully.")
            self._emit("on_survey_acquired", img)
            return img
        except Exception as e:
            self._set_state(AcquisitionState.ERROR, f"Survey acquisition failed: {e}")
            raise

    def process_survey(self) -> np.ndarray:
        """Run image filtering pipeline on survey image."""
        if self.raw_survey_image is None:
            raise RuntimeError("No survey image available to process.")

        self._set_state(AcquisitionState.ANALYZING, "Processing image pipeline...")
        try:
            processed = self.pipeline.process(self.raw_survey_image)
            self.processed_survey_image = processed
            self._set_state(AcquisitionState.IDLE, "Pipeline processing complete.")
            return processed
        except Exception as e:
            self._set_state(AcquisitionState.ERROR, f"Pipeline failed: {e}")
            raise

    def detect_features(self, method: str = "contours") -> List[Cluster]:
        """Detect and segment features from processed survey image."""
        if self.processed_survey_image is None:
            self.process_survey()

        self._set_state(AcquisitionState.ANALYZING, f"Detecting features via {method}...")
        try:
            if method == "dbscan":
                clusters = self.clusterer.segment_dbscan(
                    self.processed_survey_image, original_img=self.raw_survey_image
                )
            else:
                clusters = self.clusterer.segment_contours(
                    self.processed_survey_image, original_img=self.raw_survey_image
                )

            self.detected_clusters = clusters
            self._set_state(AcquisitionState.IDLE, f"Detected {len(clusters)} features.")
            self._emit("on_clusters_updated", clusters)
            return clusters
        except Exception as e:
            self._set_state(AcquisitionState.ERROR, f"Feature detection failed: {e}")
            raise

    def plan_scan_tiles(self, tile_size: int = 256, overlap: float = 0.10, tighten: bool = True) -> List[ScanTile]:
        """Generate optimized scan tiles from active clusters."""
        if not self.detected_clusters:
            self.detect_features()

        if self.raw_survey_image is None:
            shape = (512, 512)
        else:
            shape = self.raw_survey_image.shape

        self.planner.tile_size = tile_size
        self.planner.overlap_ratio = overlap
        curr_pos = self.microscope.get_stage_position()
        self.planner.stage_origin_nm = (curr_pos.x_nm, curr_pos.y_nm)

        tiles = self.planner.plan_all(
            self.detected_clusters,
            survey_shape=shape,
            binary_mask=self.processed_survey_image,
            tighten=tighten
        )
        self.planned_tiles = tiles
        self._emit("on_tiles_updated", tiles)
        self._log(f"Generated {len(tiles)} scan tiles across {len(self.detected_clusters)} clusters.")
        return tiles

    def set_arbitrary_correction_position(
        self,
        x_nm: Optional[float] = None,
        y_nm: Optional[float] = None,
        px_x: Optional[int] = None,
        px_y: Optional[int] = None
    ) -> None:
        """Set arbitrary stage coordinate or survey pixel for in-flight corrections."""
        if x_nm is not None and y_nm is not None:
            self.arbitrary_correction_position_nm = (float(x_nm), float(y_nm))
            self.drift_routine.arbitrary_position_nm = self.arbitrary_correction_position_nm
            self.autofocus_routine.arbitrary_position_nm = self.arbitrary_correction_position_nm

        if px_x is not None and px_y is not None:
            self.arbitrary_correction_position_px = (int(px_x), int(px_y))
            self.drift_routine.arbitrary_position_px = self.arbitrary_correction_position_px
            self.autofocus_routine.arbitrary_position_px = self.arbitrary_correction_position_px

        self._log(f"Arbitrary correction target set to: nm={self.arbitrary_correction_position_nm}, px={self.arbitrary_correction_position_px}")

    # ---------------- Automated Acquisition Queue ----------------

    def start_acquisition(self) -> None:
        """Start asynchronous execution of the planned scan tiles."""
        if not self.planned_tiles:
            raise RuntimeError("No planned scan tiles in queue. Please plan scan tiles first.")

        if self.state in (AcquisitionState.RUNNING, AcquisitionState.PAUSED):
            return

        self._stop_requested.clear()
        self._pause_requested.clear()
        self._thread = threading.Thread(target=self._run_acquisition_loop, daemon=True)
        self._thread.start()

    def pause_acquisition(self) -> None:
        """Pause the running acquisition loop after the current tile."""
        if self.state == AcquisitionState.RUNNING:
            self._pause_requested.set()
            self._set_state(AcquisitionState.PAUSED, "Paused by user.")

    def resume_acquisition(self) -> None:
        """Resume execution from paused state."""
        if self.state == AcquisitionState.PAUSED:
            self._pause_requested.clear()
            self._set_state(AcquisitionState.RUNNING, "Resuming acquisition...")

    def stop_acquisition(self) -> None:
        """Halt execution immediately."""
        self._stop_requested.set()
        if self.camera:
            try:
                self.camera.stop_acquisition()
            except Exception:
                pass
        self._set_state(AcquisitionState.STOPPED, "Acquisition stopped by user.")

    def _run_acquisition_loop(self) -> None:
        self._set_state(AcquisitionState.RUNNING, "Starting automated data collection...")
        total_tiles = len(self.planned_tiles)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        h5_file = self.save_dir / f"adc_dataset_{timestamp_str}.h5"

        try:
            for idx in range(self.current_tile_index, total_tiles):
                # Check stop request
                if self._stop_requested.is_set():
                    self._set_state(AcquisitionState.STOPPED, "Acquisition terminated.")
                    return

                # Check pause request
                while self._pause_requested.is_set():
                    time.sleep(0.2)
                    if self._stop_requested.is_set():
                        self._set_state(AcquisitionState.STOPPED, "Acquisition terminated.")
                        return

                self.current_tile_index = idx
                tile = self.planned_tiles[idx]

                self._log(f"Acquiring Tile {idx + 1}/{total_tiles} (Cluster {tile.cluster_id})")

                # In-flight pluggable corrections (Drift, Focus, Custom routines)
                fixed_cl = None
                if self.detected_clusters:
                    if self.drift_routine.fixed_cluster_id is not None:
                        matches = [c for c in self.detected_clusters if c.cluster_id == self.drift_routine.fixed_cluster_id]
                        fixed_cl = matches[0] if matches else self.detected_clusters[0]
                    else:
                        fixed_cl = self.detected_clusters[0]

                ctx = CorrectionContext(
                    scan_index=idx,
                    total_scans=total_tiles,
                    current_tile=tile,
                    previous_tile=self.planned_tiles[idx - 1] if idx > 0 else None,
                    fixed_cluster=fixed_cl,
                    arbitrary_position_nm=self.arbitrary_correction_position_nm,
                    arbitrary_position_px=self.arbitrary_correction_position_px,
                    microscope=self.microscope,
                    scanner=self.scanner,
                    survey_image=self.raw_survey_image,
                    nm_per_pixel=self.planner.nm_per_pixel
                )

                corr_results = self.corrections_manager.run_corrections_if_due(ctx)
                for cr in corr_results:
                    if cr.executed:
                        self._log(f"[{cr.routine_name.upper()} CORRECTION] {cr.message}")
                        if cr.routine_name == "drift" and (cr.dx_nm != 0 or cr.dy_nm != 0):
                            self.microscope.set_stage_relative(cr.dx_nm, cr.dy_nm)

                # Unblank beam for acquisition
                self.microscope.set_beam_blank(False)

                # Move stage or scan coils to tile location
                self.microscope.set_stage_position(tile.stage_x_nm, tile.stage_y_nm)

                # Configure and trigger camera if present
                dwell_s = 0.001
                if self.camera:
                    self.camera.configure(num_frames=100, bit_depth=12, dwell_s=dwell_s)
                    self.camera.start_acquisition()
                    # Trigger scan engine TTL
                    self.scanner.trigger_ttl(line=0, high=True)
                    time.sleep(0.05)
                    self.scanner.trigger_ttl(line=0, high=False)

                    # Fetch live sample frame
                    frame = self.camera.get_latest_frame()
                    if frame is not None:
                        self.latest_diffraction_frame = frame
                        self._emit("on_frame_received", frame)

                        # Evaluate in-flight live feedback routine
                        target = ScanTarget(
                            target_id=tile.tile_id,
                            x_px=tile.x,
                            y_px=tile.y,
                            width_px=tile.width,
                            height_px=tile.height,
                            stage_x_nm=tile.stage_x_nm,
                            stage_y_nm=tile.stage_y_nm,
                            dwell_time_s=dwell_s
                        )
                        decision = self.feedback_manager.evaluate_frame(frame, target)
                        if decision.action == DecisionAction.SKIP:
                            self._log(f"Tile {idx + 1} skipped by live feedback: {decision.reason}")
                            tile.completed = True
                            continue
                        elif decision.action == DecisionAction.ABORT:
                            self._log(f"Acquisition aborted by live feedback: {decision.reason}")
                            self._set_state(AcquisitionState.STOPPED, decision.reason)
                            return

                    self.camera.stop_acquisition()

                # Mark tile completed
                tile.completed = True
                self._emit("on_tile_progress", idx + 1, total_tiles, tile)

                # Small settling delay
                time.sleep(0.05)

            # Blanket beam after completion
            self.microscope.set_beam_blank(True)

            # Save full HDF5 container
            self._log(f"Saving acquisition archive to {h5_file}...")
            writer = HDF5Writer(h5_file)
            writer.write_dataset(
                survey_image=self.raw_survey_image,
                tiles=self.planned_tiles,
                metadata=self.microscope.get_all_parameters(),
                notes=f"Automated acquisition of {total_tiles} tiles across {len(self.detected_clusters)} clusters."
            )

            self.current_tile_index = 0
            self._set_state(AcquisitionState.COMPLETED, f"All {total_tiles} tiles collected successfully!")

        except Exception as e:
            self.microscope.set_beam_blank(True)
            self._set_state(AcquisitionState.ERROR, f"Acquisition error: {e}")
            raise

    # ---------------- Multi-Site Serial Acquisition ----------------

    def start_multi_site_acquisition(self, settling_time_s: float = 1.0) -> None:
        """Start serial acquisition across all pre-chosen stage positions."""
        if not self.multi_site_manager.sites:
            raise RuntimeError("No stage sites configured. Please record or add stage positions first.")

        if self.state in (AcquisitionState.RUNNING, AcquisitionState.PAUSED):
            return

        self._stop_requested.clear()
        self._pause_requested.clear()
        self._thread = threading.Thread(
            target=self._run_multi_site_loop,
            args=(settling_time_s,),
            daemon=True
        )
        self._thread.start()

    def _run_multi_site_loop(self, settling_time_s: float = 1.0) -> None:
        sites = self.multi_site_manager.sites
        total_sites = len(sites)
        self._log(f"Starting Multi-Site Serial Acquisition across {total_sites} stage positions...")

        try:
            for idx, site in enumerate(sites):
                if self._stop_requested.is_set():
                    site.status = SiteStatus.SKIPPED
                    self._set_state(AcquisitionState.STOPPED, "Multi-site acquisition terminated by user.")
                    return

                site.status = SiteStatus.IN_PROGRESS
                self._emit("on_multi_site_progress", idx + 1, total_sites, site)
                self._log(f"--- [Site {idx+1}/{total_sites}] Moving to {site.label} (X={site.x_nm:.1f}, Y={site.y_nm:.1f} nm) ---")

                # 1. Drive stage to target coordinates
                self.microscope.set_stage_position(site.x_nm, site.y_nm)
                time.sleep(settling_time_s)

                # Reset drift tracker for new physical area
                self.drift_tracker.reset()

                # 2. Acquire survey image
                self._log(f"Site {site.site_id}: Acquiring survey STEM...")
                survey = self.acquire_survey()

                # 3. Process image filter pipeline
                self._log(f"Site {site.site_id}: Filtering survey image...")
                self.process_survey()

                # 4. Segment features
                clusters = self.detect_features()
                site.clusters_found = len(clusters)
                self._log(f"Site {site.site_id}: Detected {len(clusters)} features.")

                if not clusters:
                    self._log(f"Site {site.site_id}: No features found. Skipping tile scans.")
                    site.status = SiteStatus.SKIPPED
                    self._emit("on_multi_site_progress", idx + 1, total_sites, site)
                    continue

                # 5. Plan scan tiles
                tiles = self.plan_scan_tiles()
                self._log(f"Site {site.site_id}: Planned {len(tiles)} scan tiles.")

                # 6. Execute tile scan queue
                self._run_acquisition_loop()
                site.tiles_collected = len(tiles)
                site.status = SiteStatus.COMPLETED
                self._emit("on_multi_site_progress", idx + 1, total_sites, site)
                self._log(f"Site {site.site_id} ({site.label}) completed successfully.")

            self._set_state(AcquisitionState.COMPLETED, f"All {total_sites} stage positions acquired successfully!")
            self._emit("on_multi_site_completed")

        except Exception as e:
            self._set_state(AcquisitionState.ERROR, f"Multi-site acquisition error: {e}")
            raise


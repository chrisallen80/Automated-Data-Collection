# Walkthrough: Modular Rewrite & Modern Browser GUI for Automated Data Collection (v2.0)

We have completely redesigned and rewritten the Automated Data Collection codebase into a clean, decoupled, highly modular architecture with a modern browser-based Web GUI and full headless execution support. The legacy PyQt5 code remains intact in `GUI/` for full backwards compatibility and reference.

---

## 1. Architecture Highlights

### Core Engine (`adc/`)
The engine is completely decoupled from any graphical framework and can run in GUI mode or headless CLI mode:
- **Hardware Abstraction Layer (`adc/hardware/`)**:
  - Unified interfaces: `BaseMicroscope`, `BaseScanner`, `BaseCamera`, `BaseEDX`.
  - Production adapters: `PyJEMAdapter` (ARM200F TEM3 stage, lens, def, ht, eos), `ScanEngineQD` (Quantum Detectors scan engine), `MerlinClient` (TCP client on ports 6341/6342), and `EDXAdapter`.
  - High-fidelity offline simulators (`adc/hardware/mock/`): `MockMicroscope`, `MockScanner`, `MockMerlin`, and `MockEDX` enabling full testing without physical microscope access.
- **Image Processing & 4D-STEM (`adc/imaging/`)**:
  - `ImagePipeline`: Configurable filter chain (Gaussian blur, sharpen, median, dual/otsu/adaptive thresholding, morphological open/close).
  - `FeatureClusterer`: Segment features and nanoparticles using either connected component Contours or DBSCAN.
  - `GridPlanner`: Overlapping tile decomposition with **bounding box tightening** (to eliminate beam time wasted scanning empty vacuum) and **snake order sorting**.
  - `STEM4DProcessor`: Virtual aperture integration (VADF, VBF, VABF), differential phase contrast / Center of Mass (COM), and Bragg peak crystallinity analysis.
- **Scanning Trajectories (`adc/scanning/`)**:
  - `PatternGenerator`: Snake, raster, Archimedean spiral, compressive sensing sparse random, and arbitrary paths.
- **In-Flight Closed-Loop Corrections (`adc/corrections/`)**:
  - `CorrectionsManager`: Central coordinator managing active routines, independent scan intervals, and execution dispatching.
  - `DriftCorrectionRoutine`: Independent enable toggle and scan frequency (every $N$ scans); configurable reference location mode (`FIXED_REGION` with anchor cluster selection, `ARBITRARY_POSITION`, `PREVIOUS_SCAN`, or `NEXT_SCAN`).
  - `AutofocusCorrectionRoutine`: Independent enable toggle and scan frequency (every $M$ scans); configurable location mode (`NEXT_SCAN`, `ARBITRARY_POSITION`, `PREVIOUS_SCAN`, `FIXED_REGION`).
  - **Arbitrary Target Support**: Allows setting an arbitrary stage $(X, Y)$ coordinate in nanometers or survey pixel coordinate $(px_x, px_y)$, driving to it for drift tracking or autofocus sweeps, and visually annotating it with a crosshair target on the survey viewer.
  - `BaseCorrectionRoutine` & `@adc_correction`: Pluggable architecture allowing arbitrary future corrections (beam tilt, stigmation, temperature drift) to be dropped in seamlessly.
  - `EmissionMonitor`: FEG emission stability tracking and flashing alert trigger.
- **Extensible Plugin System (`adc/plugins/`)**:
  - Dynamic loader allowing users to import standalone Python scripts or Jupyter-exported functions decorated with `@adc_filter` (custom thresholding and filter steps), `@adc_clustering`, `@adc_decision`, `@adc_live_feedback`, or `@adc_correction`.
  - Built-in routines for DBSCAN, high-contrast decision ranking, and adaptive dwell time steering.
- **Acquisition Orchestrator & Archival (`adc/acquisition/`)**:
  - `AcquisitionEngine`: Threaded state machine (`IDLE`, `SURVEYING`, `RUNNING`, `PAUSED`, `STOPPED`, `COMPLETED`).
  - `MultiSiteManager` & `StageSite`: Preselect and record multiple stage positions, drive to saved positions, and execute automated serial batch acquisitions.
  - `LiveFeedbackManager`: In-flight evaluation of diffraction frames to dynamically steer scan queues.
  - `HDF5Writer`: NeXus-compliant container archiving survey micrographs, tile coordinates, column metadata, and virtual detector images.

---

## 2. Modern Browser User Interface (`web_gui/`)

Built with **NiceGUI**, Quasar components, and dark electron microscopy styling:
- **Interactive Micrograph Survey Viewer**: Real-time canvas with interactive crosshairs; one-click image targeting allows clicking anywhere on the survey micrograph to immediately lock that coordinate as the arbitrary drift/focus reference anchor.
- **Stage Navigator Tab**: Live stage coordinates readout, jog buttons, one-click **"Record Current Position"**, interactive waypoints table with **"Drive Here"** and delete buttons, and **"Run Multi-Site Batch"** controls with progress monitoring.
- **Survey & Pipeline Tab**: Real-time filter tuning (Gaussian blur, sharpen, dual threshold, morphology) with live side-by-side preview; dynamic **Threshold Method** dropdown populated with both built-in methods and loaded custom plugins (`@adc_filter`), plus an **Active Custom Plugins** list with one-click activation.
- **Feature Clusters Tab**: Contour / DBSCAN segmentation, cluster list table with sorting, and checkbox selection.
- **Scan Grid & Queue Tab**: Tile size, overlap slider, snake trajectory planning, in-flight corrections quick controls, and start / pause / resume / stop execution controls with live progress bar.
- **4D-STEM & Merlin Tab**: Live diffraction pattern stream, virtual aperture radius sliders (VBF, VADF), live Center of Mass indicator, and Bragg crystallinity index meter.
- **Hardware & Corrections Tab**: Real-time stage telemetry, OLf defocus, HT, beam blank switch, manual stage jog, independent drift tracking frequency & location controls (including Arbitrary Position with "Use Current Stage Pos" and manual $X/Y$ inputs), independent autofocus frequency & location controls, and external Python script loader.
- **Experiment Activity Log**: Collapsible rolling terminal log capturing engine events in real time.

---

## 3. Verification & Testing

### Automated Unit Test Suite
Ran full test suite covering all 13 major subcomponents across `test_adc_core.py`, `test_multi_site.py`, and `test_corrections.py`:
- `test_mock_hardware`: Stage coordinates, beam blanking, defocus, and synthetic frame generation.
- `test_image_pipeline_and_clustering`: Filter pipeline and multi-particle centroid detection.
- `test_grid_planner`: Tile decomposition, overlap calculation, and boundary clamping.
- `test_stem4d_processing`: VBF/VADF integration, COM phase contrast, and Bragg reflection detection.
- `test_drift_tracker`: Sub-pixel shift measurement using phase cross-correlation.
- `test_autofocus`: OLf sweep and quadratic peak optimization.
- `test_plugin_dynamic_loading`: Dynamic loading and execution of user routine from an external file.
- `test_full_acquisition_engine_workflow`: End-to-end automated multi-tile acquisition with HDF5 file generation.
- `test_multi_site_manager`: Stage position recording, waypoint table, and JSON save/load.
- `test_multi_site_batch_execution`: Automated batch loop driving to multiple stage positions, surveying, clustering, and collecting tile scans.
- `test_independent_selection_and_intervals`: Independent drift / autofocus enable toggles and scan intervals ($N$ and $M$ scans).
- `test_reference_location_modes`: Drift and focus execution across `FIXED_REGION`, `ARBITRARY_POSITION` (stage nm and survey pixels), `PREVIOUS_SCAN`, and `NEXT_SCAN` strategies.
- `test_pluggable_custom_correction_routine`: Creation, registration, and automatic interval dispatch of a custom in-flight correction.

**Result**: `ALL 13 TESTS PASSED!`

### Headless CLI Verification
Executed `python run_app.py --headless`:
- Successfully initialized mock hardware, acquired survey, segmented clusters, planned 3 scan tiles, ran the acquisition loop, and produced a verified NeXus HDF5 file: `acquisitions/adc_dataset_20260923_233733.h5`.

### Web Server Verification
Executed `python run_app.py --port 8085` and sent HTTP request:
- Received `HTTP 200 OK` with full HTML and Quasar/NiceGUI UI bundle.

---

## 4. How to Run

### In Browser GUI Mode (Default: Mock Simulator)
```powershell
python run_app.py
```
Then navigate to: **`http://localhost:8080`**

### With Real Microscope Hardware (Online ARM200F)
```powershell
python run_app.py --online --merlin-ip 192.168.1.10 --merlin-port 6341
```

### In Headless CLI Mode (No Browser)
```powershell
python run_app.py --headless --save-dir ./acquisitions
```

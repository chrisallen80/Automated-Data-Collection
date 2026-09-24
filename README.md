# Automated Data Collection (ADC) v2.0

Modular, decoupled control engine and modern browser-based web GUI for automated high-resolution 4D-STEM and electron microscopy data acquisition on JEOL ARM200F microscopes.

---

## Architecture Overview

ADC v2.0 separates the core microscopy logic from the presentation layer:

```
Automated-Data-Collection/
├── adc/                         # Pure Headless Core Engine
│   ├── hardware/                # Hardware Abstraction Layer (HAL)
│   │   ├── base.py              # BaseMicroscope, BaseScanner, BaseCamera, BaseEDX
│   │   ├── pyjem_adapter.py     # Production JEOL ARM200F PyJEM adapter
│   │   ├── scan_engine_qd.py    # Quantum Detectors ScanEngine STEM adapter
│   │   ├── merlin_client.py     # Merlin Medipix TCP client (ports 6341 / 6342)
│   │   ├── edx_adapter.py       # Pluggable EDX spectrometer adapter
│   │   └── mock/                # High-fidelity offline hardware simulators
│   ├── imaging/                 # Micrograph Preprocessing & 4D-STEM
│   │   ├── pipeline.py          # Configurable filter chain (Blur, Sharpen, Dual Threshold, Morphology)
│   │   ├── clustering.py        # Feature segmentation (Contours, DBSCAN)
│   │   ├── grid_planner.py      # Overlapping tile decomposition, bbox tightening & snake ordering
│   │   └── stem4d.py            # Virtual detectors (VADF, VBF, VABF), DPC / COM, Bragg crystallinity
│   ├── scanning/                # Scan Trajectory Patterns
│   │   └── patterns.py          # Snake, Raster, Spiral, Sparse Random, Arbitrary paths
│   ├── corrections/             # In-Flight Closed-Loop Corrections
│   │   ├── drift.py             # Phase cross-correlation with sub-pixel registration
│   │   ├── autofocus.py         # Objective Lens Fine (OLf) sweep & quadratic peak fitting
│   │   └── emission.py          # Cold-FEG emission stability monitor & flashing alerts
│   ├── plugins/                 # Extensible Custom Routines System
│   │   ├── base.py              # Protocols for custom clustering, decision, & live feedback
│   │   ├── loader.py            # Dynamic loader for external .py user scripts
│   │   └── builtins/            # Built-in sample routines (DBSCAN, Bragg decision, Adaptive dwell)
│   └── acquisition/             # Orchestrator & Archival
│       ├── engine.py            # Acquisition state machine (IDLE, RUNNING, PAUSED, etc.)
│       ├── live_feedback.py     # In-flight frame analyzer & queue steering
│       └── hdf5_writer.py       # NeXus-compliant HDF5 archival container
├── web_gui/                     # Modern Browser User Interface (NiceGUI)
│   ├── app.py                   # Main reactive web layout and navigation
│   ├── state.py                 # Shared UI state & base64 image streaming
│   └── components/              # Modular UI components (Canvas, Pipeline, Cluster, Scan, STEM, Corrections)
├── GUI/                         # Legacy PyQt5 application (preserved for reference)
├── tests/                       # Automated unit tests
└── run_app.py                   # Root launcher CLI script
```

---

## Quick Start

### 1. Launch Browser Web GUI (Default: Mock Simulation)

```bash
python run_app.py
```
Open your browser to: **`http://localhost:8080`**

### 2. Connect to Real Microscope Hardware (ARM200F / PyJEM / Merlin)

```bash
python run_app.py --online --merlin-ip 192.168.1.10 --merlin-port 6341
```

### 3. Run Headless Script (No Browser)

```bash
python run_app.py --headless --save-dir ./acquisitions
```

---

---

## Writing & Plugging in Custom Algorithms (Plugins)

Researchers can extend the acquisition and image analysis pipeline by writing standalone Python scripts. The system dynamically loads functions decorated with ADC decorators:

### 1. Custom Image Filtering & Thresholding (`@adc_filter`)
Create custom binarization or contrast algorithms (e.g., Sauvola, Yen, or machine-learning segmenters):

```python
# custom_threshold_plugin.py
import cv2
import numpy as np
from skimage.filters import threshold_sauvola, threshold_yen
from adc.plugins import adc_filter

@adc_filter
def sauvola_threshold(img: np.ndarray, params: dict) -> np.ndarray:
    """Custom local adaptive Sauvola thresholding for non-uniform background."""
    window_size = int(params.get("window_size", 25))
    k = float(params.get("k", 0.2))
    thresh_map = threshold_sauvola(img, window_size=window_size, k=k)
    return (img > thresh_map).astype(np.uint8) * 255

@adc_filter
def yen_threshold(img: np.ndarray, params: dict) -> np.ndarray:
    """Entropy-based Yen thresholding."""
    val = threshold_yen(img)
    return (img > val).astype(np.uint8) * 255
```

### 2. Custom Feature Clustering (`@adc_clustering`)
```python
from adc.plugins import adc_clustering
from adc.imaging.clustering import Cluster

@adc_clustering
def my_watershed_clusterer(binary_mask: np.ndarray, config: dict):
    # Custom watershed or HDBSCAN logic
    ...
    return [Cluster(...), ...]
```

### 3. Custom In-Flight Closed-Loop Feedback (`@adc_live_feedback`)
```python
from adc.plugins import adc_live_feedback, LiveDecision, DecisionAction
from adc.imaging.stem4d import STEM4DProcessor

@adc_live_feedback
def my_adaptive_dwell(diffraction_frame: np.ndarray, target, config: dict) -> LiveDecision:
    # Dynamically adjust dwell time or skip amorphous carbon
    cryst = STEM4DProcessor.compute_crystallinity_index(diffraction_frame)
    if cryst["is_crystalline"]:
        return LiveDecision(action=DecisionAction.ADJUST_DWELL, new_dwell_s=0.005)
    return LiveDecision(action=DecisionAction.ACQUIRE)
```

---

## Activating Plugins in the Browser GUI

1. **Load the Script**: Go to the **Microscope & Plugins** tab (Tab 5), enter the path to your `.py` file, and click **Load Plugin Script**.
2. **Select in Pipeline**:
   - Navigate to the **Preprocessing & Filters** tab (Tab 1).
   - Your custom algorithm (e.g. `sauvola_threshold`) will automatically appear in the **Threshold Method** dropdown.
   - Or click **Select Method** in the **Active Pipeline Filters & Custom Plugins** list.
3. **Instant Preview**: Clicking **Process Survey** immediately processes the survey micrograph using your custom routine.

---

## Running Tests

Execute the unit test suite:

```bash
python -c "from tests.test_adc_core import *; test_mock_hardware(); test_image_pipeline_and_clustering(); test_grid_planner(); test_stem4d_processing(); test_drift_tracker(); test_autofocus(); test_plugin_dynamic_loading(); test_full_acquisition_engine_workflow(); print('All tests passed!')"
```
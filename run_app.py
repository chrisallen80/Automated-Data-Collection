"""
Entry point for Automated Data Collection (ADC) v2.0.
Supports browser-based web interface and headless execution modes.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adc.hardware.mock import MockMicroscope, MockScanner, MockMerlin, MockEDX
from adc.acquisition.engine import AcquisitionEngine


def main():
    parser = argparse.ArgumentParser(description="Automated Data Collection (ADC) - Modular STEM Controller")
    parser.add_argument("--online", action="store_true", help="Connect to physical microscope and detectors (PyJEM, Merlin, ScanEngine)")
    parser.add_argument("--mock", action="store_true", default=None, help="Use offline hardware simulation (default if online is not set)")
    parser.add_argument("--merlin-ip", type=str, default="127.0.0.1", help="Merlin TCP camera IP address")
    parser.add_argument("--merlin-port", type=int, default=6341, help="Merlin TCP camera port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web GUI host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Web GUI port (default: 8080)")
    parser.add_argument("--headless", action="store_true", help="Run automated acquisition in headless mode without browser UI")
    parser.add_argument("--save-dir", type=str, default="./acquisitions", help="Directory for output HDF5 datasets")

    args = parser.parse_args()

    use_mock = True
    if args.online:
        use_mock = False

    print("=" * 65)
    print("  Automated Data Collection (ADC) v2.0")
    print(f"  Mode: {'MOCK SIMULATOR' if use_mock else 'ONLINE HARDWARE'}")
    print(f"  Save Directory: {args.save_dir}")
    print("=" * 65)

    if use_mock:
        scope = MockMicroscope()
        scanner = MockScanner(microscope=scope)
        camera = MockMerlin(detector_size=256)
        edx = MockEDX()
    else:
        try:
            from adc.hardware.pyjem_adapter import PyJEMAdapter
            from adc.hardware.scan_engine_qd import ScanEngineQD
            from adc.hardware.merlin_client import MerlinClient
            from adc.hardware.edx_adapter import EDXAdapter

            print("Connecting to JEOL PyJEM...")
            scope = PyJEMAdapter()
            scope.connect()

            print("Connecting to ScanEngine...")
            scanner = ScanEngineQD()
            scanner.connect()

            print(f"Connecting to Merlin detector at {args.merlin_ip}:{args.merlin_port}...")
            camera = MerlinClient(host=args.merlin_ip, cmd_port=args.merlin_port)
            camera.connect()

            edx = EDXAdapter()
        except Exception as e:
            print(f"\n[Hardware Connection Error] Could not connect to real microscope hardware: {e}")
            print("Falling back to offline MOCK mode for simulation.\n")
            scope = MockMicroscope()
            scanner = MockScanner(microscope=scope)
            camera = MockMerlin(detector_size=256)
            edx = MockEDX()

    engine = AcquisitionEngine(
        microscope=scope,
        scanner=scanner,
        camera=camera,
        edx=edx,
        save_dir=args.save_dir
    )

    if args.headless:
        print("\nStarting headless automated acquisition workflow...")
        engine.acquire_survey()
        engine.process_survey()
        engine.detect_features()
        tiles = engine.plan_scan_tiles(tile_size=256, overlap=0.10)
        print(f"Queueing {len(tiles)} tiles for acquisition...")
        engine._run_acquisition_loop()
        print("Headless acquisition complete!")
    else:
        from web_gui import start_server
        print(f"\nLaunching Browser GUI at http://{args.host}:{args.port}")
        start_server(engine, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

"""
Dynamic Plugin Loader for external Python scripts and routines.
"""

from __future__ import annotations
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from .base import ClusteringFunc, DecisionFunc, LiveFeedbackFunc


class PluginRegistry:
    """
    Central repository of available clustering routines, decision makers,
    custom filter/thresholding algorithms, and live in-flight feedback analyzers.
    """

    def __init__(self):
        self.filter_routines: Dict[str, Any] = {}
        self.clustering_routines: Dict[str, ClusteringFunc] = {}
        self.decision_routines: Dict[str, DecisionFunc] = {}
        self.live_feedback_routines: Dict[str, LiveFeedbackFunc] = {}
        self.correction_routines: Dict[str, Any] = {}
        self.loaded_modules: Dict[str, Any] = {}

    def register_filter(self, name: str, func: Any) -> None:
        self.filter_routines[name] = func

    def register_clustering(self, name: str, func: ClusteringFunc) -> None:
        self.clustering_routines[name] = func

    def register_decision(self, name: str, func: DecisionFunc) -> None:
        self.decision_routines[name] = func

    def register_live_feedback(self, name: str, func: LiveFeedbackFunc) -> None:
        self.live_feedback_routines[name] = func

    def register_correction(self, name: str, routine_cls_or_obj: Any) -> None:
        self.correction_routines[name] = routine_cls_or_obj

    def load_from_file(self, file_path: str | Path) -> Dict[str, List[str]]:
        """
        Dynamically import an external Python file and automatically register
        any routine functions annotated or decorated for ADC.
        Returns a dictionary listing newly discovered routine names by category.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Plugin script not found: {path}")

        module_name = f"adc_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Python module spec from: {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.loaded_modules[path.stem] = module

        discovered = {
            "filter": [],
            "clustering": [],
            "decision": [],
            "live_feedback": [],
            "correction": []
        }

        # Inspect all callables in module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not callable(attr) or attr_name.startswith("_"):
                continue

            # Check explicit convention or attribute tags
            if getattr(attr, "_is_adc_filter", False) or attr_name.startswith("filter_") or attr_name.startswith("thresh_"):
                self.register_filter(attr_name, attr)
                discovered["filter"].append(attr_name)

            elif getattr(attr, "_is_adc_clustering", False) or attr_name.startswith("cluster_"):
                self.register_clustering(attr_name, attr)
                discovered["clustering"].append(attr_name)

            elif getattr(attr, "_is_adc_decision", False) or attr_name.startswith("decide_"):
                self.register_decision(attr_name, attr)
                discovered["decision"].append(attr_name)

            elif getattr(attr, "_is_adc_live_feedback", False) or attr_name.startswith("feedback_"):
                self.register_live_feedback(attr_name, attr)
                discovered["live_feedback"].append(attr_name)

            elif getattr(attr, "_is_adc_correction", False) or attr_name.startswith("correct_"):
                self.register_correction(attr_name, attr)
                discovered["correction"].append(attr_name)

        return discovered


# Decorators for user convenience in external scripts
def adc_filter(func: Any) -> Any:
    func._is_adc_filter = True
    return func


def adc_clustering(func: ClusteringFunc) -> ClusteringFunc:
    func._is_adc_clustering = True
    return func


def adc_decision(func: DecisionFunc) -> DecisionFunc:
    func._is_adc_decision = True
    return func


def adc_live_feedback(func: LiveFeedbackFunc) -> LiveFeedbackFunc:
    func._is_adc_live_feedback = True
    return func


def adc_correction(obj: Any) -> Any:
    obj._is_adc_correction = True
    return obj


# Global singleton instance
GLOBAL_REGISTRY = PluginRegistry()

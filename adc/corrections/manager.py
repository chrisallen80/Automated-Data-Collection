"""
Corrections Manager for orchestrating independent and pluggable in-flight corrections.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from .base import BaseCorrectionRoutine, CorrectionContext, CorrectionResult


class CorrectionsManager:
    """
    Central coordinator for in-flight microscope corrections.
    Manages active routines, monitors scan intervals, and dispatches executions.
    """

    def __init__(self):
        self._routines: Dict[str, BaseCorrectionRoutine] = {}

    def register(self, routine: BaseCorrectionRoutine) -> None:
        """Register or replace a correction routine."""
        self._routines[routine.name.lower()] = routine

    def unregister(self, name: str) -> bool:
        """Remove a routine by name."""
        return self._routines.pop(name.lower(), None) is not None

    def get(self, name: str) -> Optional[BaseCorrectionRoutine]:
        """Retrieve routine by name."""
        return self._routines.get(name.lower())

    def all_routines(self) -> List[BaseCorrectionRoutine]:
        """Return all registered correction routines."""
        return list(self._routines.values())

    def run_corrections_if_due(self, context: CorrectionContext) -> List[CorrectionResult]:
        """
        Check all registered corrections and execute any that are due based on
        their independent scan interval and enabled status.
        """
        results: List[CorrectionResult] = []

        for name, routine in self._routines.items():
            if routine.should_run(context):
                try:
                    res = routine.execute(context)
                    routine.last_run_scan_index = context.scan_index
                    results.append(res)
                except Exception as e:
                    results.append(CorrectionResult(
                        routine_name=routine.name,
                        executed=False,
                        message=f"Error executing {routine.name}: {e}"
                    ))

        return results

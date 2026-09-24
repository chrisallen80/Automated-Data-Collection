"""
Live Feedback Manager: In-flight analysis and dynamic steering of the acquisition queue.
"""

from __future__ import annotations
import queue
import threading
from typing import Optional, Callable, Dict, Any
import numpy as np
from ..plugins.base import ScanTarget, LiveDecision, DecisionAction, LiveFeedbackFunc


class LiveFeedbackManager:
    """
    Evaluates incoming 4D-STEM frames or tile previews against active live feedback routines.
    Directs the acquisition engine to adjust dwell time or skip tiles on the fly.
    """

    def __init__(self, feedback_routine: Optional[LiveFeedbackFunc] = None, config: Optional[Dict[str, Any]] = None):
        self.feedback_routine = feedback_routine
        self.config = config or {}
        self._history: list[LiveDecision] = []

    def set_routine(self, routine: Optional[LiveFeedbackFunc], config: Optional[Dict[str, Any]] = None) -> None:
        self.feedback_routine = routine
        if config is not None:
            self.config = config

    def evaluate_frame(self, frame: np.ndarray, target: ScanTarget) -> LiveDecision:
        """
        Evaluate an acquired frame and determine in-flight steering action.
        """
        if self.feedback_routine is None:
            return LiveDecision(action=DecisionAction.ACQUIRE, reason="Default: No feedback routine active.")

        try:
            decision = self.feedback_routine(frame, target, self.config)
            self._history.append(decision)
            return decision
        except Exception as e:
            return LiveDecision(
                action=DecisionAction.ACQUIRE,
                reason=f"Error executing live feedback routine: {e}"
            )

    @property
    def history(self) -> list[LiveDecision]:
        return list(self._history)

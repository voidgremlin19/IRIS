"""
temporal.py — Temporal Buffer
Maintains a rolling context window (~6 seconds) for aggregating
analysis results across multiple frames/inputs.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from collections import deque

from src.config import TEMPORAL_BUFFER_SECONDS, TEMPORAL_MAX_FRAMES

logger = logging.getLogger(__name__)


class TemporalBuffer:
    """
    Maintains a sliding window of analysis results for temporal consistency.
    Aggregates decisions over a configurable time window to produce
    stable, context-aware warnings.
    """

    def __init__(
        self,
        buffer_seconds: float = TEMPORAL_BUFFER_SECONDS,
        max_frames: int = TEMPORAL_MAX_FRAMES,
    ):
        self.buffer_seconds = buffer_seconds
        self.max_frames = max_frames
        self.buffer: deque = deque(maxlen=max_frames)

    def add(self, analysis_result: dict) -> None:
        """
        Add a new analysis result to the buffer.
        
        Args:
            analysis_result: Complete pipeline output for one frame
        """
        entry = {
            "timestamp": time.time(),
            "result": analysis_result,
        }
        self.buffer.append(entry)
        self._cleanup_old_entries()

    def _cleanup_old_entries(self) -> None:
        """Remove entries older than the buffer window."""
        current_time = time.time()
        cutoff = current_time - self.buffer_seconds

        while self.buffer and self.buffer[0]["timestamp"] < cutoff:
            self.buffer.popleft()

    def get_aggregated_warning(self) -> dict:
        """
        Aggregate analysis results within the temporal window
        to produce a stable warning.
        
        Returns:
            Aggregated warning with consensus decision and risk level
        """
        self._cleanup_old_entries()

        if not self.buffer:
            return {
                "warning": "No recent data available.",
                "decision": "proceed_normally",
                "risk": "none",
                "confidence": 0.0,
                "frame_count": 0,
                "window_seconds": 0.0,
            }

        # Collect all decisions and risks
        decisions = []
        risks = []
        alerts = []
        all_objects = {}

        for entry in self.buffer:
            result = entry.get("result", {})
            reasoning = result.get("reasoning", {})

            if reasoning:
                decisions.append(reasoning.get("decision", "proceed_normally"))
                risks.append(reasoning.get("risk", "none"))
                alert = reasoning.get("alert", "")
                if alert:
                    alerts.append(alert)

                # Aggregate object counts
                obj_summary = reasoning.get("object_summary", {})
                for obj_type, count in obj_summary.items():
                    all_objects[obj_type] = max(
                        all_objects.get(obj_type, 0), count
                    )

        # ─── Consensus Decision ──────────────────────────────────
        decision_priority = {
            "emergency_brake": 5,
            "emergency_caution": 4,
            "extreme_caution": 4,
            "slow_down": 3,
            "maintain_lane": 2,
            "caution": 2,
            "proceed_normally": 0,
        }

        risk_priority = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
            "none": 0,
        }

        # Use highest-priority decision in the window
        if decisions:
            consensus_decision = max(
                decisions, key=lambda d: decision_priority.get(d, 0)
            )
        else:
            consensus_decision = "proceed_normally"

        # Use highest risk in the window
        if risks:
            consensus_risk = max(
                risks, key=lambda r: risk_priority.get(r, 0)
            )
        else:
            consensus_risk = "none"

        # Calculate decision consistency (confidence)
        if decisions:
            most_common = max(set(decisions), key=decisions.count)
            confidence = decisions.count(most_common) / len(decisions)
        else:
            confidence = 0.0

        # Build aggregated warning
        frame_count = len(self.buffer)
        time_span = (
            self.buffer[-1]["timestamp"] - self.buffer[0]["timestamp"]
            if frame_count > 1 else 0.0
        )

        # Select most critical alert
        primary_alert = alerts[-1] if alerts else "No active warnings."

        # Build persistent warning if risk has been consistent
        if confidence > 0.6 and consensus_risk in ("high", "critical"):
            warning = (
                f"PERSISTENT WARNING: {consensus_decision.replace('_', ' ').title()}! "
                f"Risk level: {consensus_risk}. "
                f"Consistent across {frame_count} frames "
                f"over {time_span:.1f}s."
            )
        else:
            warning = primary_alert

        return {
            "warning": warning,
            "decision": consensus_decision,
            "risk": consensus_risk,
            "confidence": round(confidence, 2),
            "frame_count": frame_count,
            "window_seconds": round(time_span, 1),
            "primary_alert": primary_alert,
            "all_alerts": list(set(alerts)),
            "detected_objects": all_objects,
        }

    def clear(self) -> None:
        """Clear the temporal buffer."""
        self.buffer.clear()

    @property
    def size(self) -> int:
        """Current number of entries in the buffer."""
        return len(self.buffer)

    @property
    def is_active(self) -> bool:
        """Whether the buffer has recent data."""
        self._cleanup_old_entries()
        return len(self.buffer) > 0

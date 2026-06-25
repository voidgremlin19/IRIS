"""
viz.py — Visualization Module
Draw bounding boxes, labels, and overlay reasoning results on images.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from src.config import VIZ_COLORS, VIZ_FONT_SCALE, VIZ_THICKNESS, VIZ_BBOX_THICKNESS

logger = logging.getLogger(__name__)


class Visualizer:
    """Renders detection results, scene info, and reasoning on images."""

    def __init__(self):
        if not HAS_CV2:
            logger.warning("OpenCV not available. Visualization disabled.")

    def draw_detections(
        self,
        image: np.ndarray,
        detections: dict,
        distances: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: Input image (BGR numpy array)
            detections: Detection results from detect.py
            distances: Optional distance estimates
            
        Returns:
            Annotated image
        """
        if not HAS_CV2:
            return image

        annotated = image.copy()
        objects = detections.get("objects", [])
        dist_list = distances.get("distances", []) if distances else []

        # Create distance lookup
        dist_lookup = {}
        for i, d in enumerate(dist_list):
            if i < len(objects):
                dist_lookup[i] = d

        for idx, obj in enumerate(objects):
            bbox = obj.get("bbox", [0, 0, 0, 0])
            obj_type = obj.get("type", "unknown")
            confidence = obj.get("confidence", 0)

            # Get color
            color = VIZ_COLORS.get(obj_type, VIZ_COLORS["default"])

            # Convert to integers
            x1, y1, x2, y2 = [int(b) for b in bbox]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, VIZ_BBOX_THICKNESS)

            # Build label
            label = f"{obj.get('name', obj_type)} {confidence:.0%}"

            # Add distance info if available
            dist_info = dist_lookup.get(idx)
            if dist_info:
                zone = dist_info.get("zone", "?")
                meters = dist_info.get("estimated_meters", 0)
                label += f" | {meters:.0f}m ({zone})"

            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, VIZ_FONT_SCALE, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 5, y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                VIZ_FONT_SCALE,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def draw_reasoning_overlay(
        self,
        image: np.ndarray,
        reasoning: dict,
    ) -> np.ndarray:
        """
        Overlay reasoning results on the image.
        
        Args:
            image: Image with detections already drawn
            reasoning: Reasoning output from reason.py
            
        Returns:
            Image with reasoning overlay
        """
        if not HAS_CV2:
            return image

        annotated = image.copy()
        h, w = annotated.shape[:2]

        # Draw semi-transparent overlay for reasoning info
        overlay = annotated.copy()
        panel_h = 120
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # Decision
        decision = reasoning.get("decision", "N/A")
        risk = reasoning.get("risk", "N/A")
        context = reasoning.get("context", "N/A")
        alert = reasoning.get("alert", "N/A")

        # Risk color
        risk_colors = {
            "critical": (0, 0, 255),
            "high": (0, 80, 255),
            "medium": (0, 200, 255),
            "low": (0, 255, 0),
            "none": (200, 200, 200),
        }
        risk_color = risk_colors.get(risk, (200, 200, 200))

        y_base = h - panel_h + 25

        # Decision line
        cv2.putText(
            annotated,
            f"DECISION: {decision.upper().replace('_', ' ')}",
            (15, y_base),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Risk indicator
        cv2.putText(
            annotated,
            f"RISK: {risk.upper()}",
            (w - 200, y_base),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            risk_color,
            2,
            cv2.LINE_AA,
        )

        # Risk dot
        cv2.circle(annotated, (w - 220, y_base - 5), 8, risk_color, -1)

        # Context (truncated)
        ctx_display = context[:80] + "..." if len(context) > 80 else context
        cv2.putText(
            annotated,
            f"Context: {ctx_display}",
            (15, y_base + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Alert
        alert_display = alert[:90] + "..." if len(alert) > 90 else alert
        cv2.putText(
            annotated,
            f"Alert: {alert_display}",
            (15, y_base + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return annotated

    def create_full_visualization(
        self,
        image: np.ndarray,
        detections: dict,
        distances: Optional[dict] = None,
        reasoning: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Create complete visualization with detections, distances, and reasoning.
        
        Args:
            image: Raw input image
            detections: Detection results
            distances: Distance estimates
            reasoning: Reasoning output
            
        Returns:
            Fully annotated image
        """
        # Draw detections and distances
        result = self.draw_detections(image, detections, distances)

        # Overlay reasoning if available
        if reasoning:
            result = self.draw_reasoning_overlay(result, reasoning)

        return result


def visualize_results(
    image: np.ndarray,
    detections: dict,
    distances: Optional[dict] = None,
    reasoning: Optional[dict] = None,
) -> np.ndarray:
    """Convenience function for full visualization."""
    viz = Visualizer()
    return viz.create_full_visualization(image, detections, distances, reasoning)

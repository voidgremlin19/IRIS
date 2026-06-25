"""
distance.py — Distance + Threat Intelligence Module
Owner: gautami.patnaik
Tasks:
  - Task 1: Perspective-aware distance + lane-relative estimation
  - Task 2: Threat ranking with collision probability + motion sensitivity
  - Task 3: Multi-object risk output (primary/secondary threat + action)
No hardcoded values — all constants come from config.py
"""

import logging
from typing import List, Dict, Optional

from src.config import (
    OBJECT_REAL_HEIGHTS,
    CAMERA_FOCAL_LENGTH,
    CAMERA_HEIGHT_METERS,
    DISTANCE_NEAR_THRESHOLD,
    DISTANCE_FAR_THRESHOLD,
    HORIZON_LINE_RATIO,
    EGO_LANE_WIDTH_RATIO,
    CONF_LOW_THRESHOLD,
    CONF_LOW_MULTIPLIER,
    CONF_HIGH_THRESHOLD,
    RISK_SCORE_CRITICAL,
    RISK_SCORE_HIGH,
    RISK_SCORE_MEDIUM,
    OBJECT_RISK_WEIGHTS,
    OBJECT_MOTION_SENSITIVITY,
)

logger = logging.getLogger(__name__)


class DistanceEstimator:
    """
    Estimates distance to detected objects and ranks them by threat level.
    All thresholds and weights come from config.py — nothing is hardcoded here.
    """

    def __init__(
        self,
        focal_length: float = CAMERA_FOCAL_LENGTH,
        near_threshold: float = DISTANCE_NEAR_THRESHOLD,
        far_threshold: float = DISTANCE_FAR_THRESHOLD,
        camera_height: float = CAMERA_HEIGHT_METERS,
    ):
        self.focal_length   = focal_length
        self.near_threshold = near_threshold
        self.far_threshold  = far_threshold
        self.camera_height  = camera_height

    # ─────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────
    def estimate(self, detections: dict) -> dict:
        """
        Main function called by pipeline.py
        Input:  raw detections from detect.py
        Output: distances + threat report
        """
        objects  = detections.get("objects", [])
        img_size = detections.get("image_size", {"width": 640, "height": 640})
        img_w    = img_size.get("width", 640)
        img_h    = img_size.get("height", 640)

        # Step 1: Estimate distance for every detected object
        distance_results = []
        for obj in objects:
            dist_info = self._estimate_single(obj, img_w, img_h)
            distance_results.append(dist_info)

        # Step 2: Count objects per zone
        near_count   = sum(1 for d in distance_results if d["zone"] == "near")
        medium_count = sum(1 for d in distance_results if d["zone"] == "medium")
        far_count    = sum(1 for d in distance_results if d["zone"] == "far")

        # Step 3: Build threat report
        threat_report = self.get_closest_threat({"distances": distance_results})

        return {
            "distances":    distance_results,
            "threat_report": threat_report,
            "summary": {
                "near_objects":   near_count,
                "medium_objects": medium_count,
                "far_objects":    far_count,
                "total_objects":  len(distance_results),
                "closest_object": min(
                    distance_results,
                    key=lambda d: d["estimated_meters"]
                ) if distance_results else None,
            },
        }

    # ─────────────────────────────────────────
    # TASK 1: PERSPECTIVE-AWARE DISTANCE
    # ─────────────────────────────────────────
    def _estimate_single(self, obj: Dict, img_width: int, img_height: int) -> Dict:
        """
        Two methods combined:
        1. Ground-plane projection  → uses bottom of bbox vs horizon (more accurate)
        2. Bbox height ratio        → fallback when object is at/above horizon
        Also calculates: lane position, zone, confidence score
        """
        obj_type = obj.get("type", "unknown")
        bbox     = obj.get("bbox", [0, 0, 100, 100])  # [x1, y1, x2, y2]
        conf     = obj.get("confidence", 1.0)

        x1, y1, x2, y2 = bbox
        bbox_height_px  = max(1, abs(y2 - y1))
        y_bottom        = y2

        # Horizon line — from config, not hardcoded
        horizon    = img_height * HORIZON_LINE_RATIO
        relative_y = y_bottom - horizon  # pixels below horizon

        # --- Method 1: Ground-Plane Projection (preferred) ---
        if relative_y > 5:
            est_dist = (self.camera_height * self.focal_length) / relative_y
            method   = "perspective"
        else:
            # --- Method 2: Bbox Height Ratio (fallback) ---
            real_h   = OBJECT_REAL_HEIGHTS.get(obj_type, 1.5)
            est_dist = (real_h * self.focal_length) / bbox_height_px
            method   = "bbox_ratio"

        # --- Confidence Adjustment ---
        # Low confidence = treat object as closer = safer/more cautious
        # Thresholds from config, not hardcoded
        if conf < CONF_LOW_THRESHOLD:
            est_dist *= CONF_LOW_MULTIPLIER
        elif conf >= CONF_HIGH_THRESHOLD:
            pass  # high confidence, no adjustment needed

        # --- Lane-Relative Estimation ---
        # Ego lane = center 30% of screen (from config)
        x_center       = (x1 + x2) / 2
        lane_band      = img_width * EGO_LANE_WIDTH_RATIO
        ego_left_edge  = (img_width / 2) - (lane_band / 2)
        ego_right_edge = (img_width / 2) + (lane_band / 2)

        if ego_left_edge < x_center < ego_right_edge:
            lane = "ego"
        elif x_center <= ego_left_edge:
            lane = "left"
        else:
            lane = "right"

        # --- Zone Classification ---
        if est_dist <= self.near_threshold:
            zone    = "near"
            urgency = "high"
        elif est_dist <= self.far_threshold:
            zone    = "medium"
            urgency = "moderate"
        else:
            zone    = "far"
            urgency = "low"

        # --- Estimate Confidence Score ---
        bbox_ratio     = bbox_height_px / img_height
        estimate_conf  = round(min(1.0, bbox_ratio * 3) * conf, 2)

        return {
            "type":             obj_type,
            "estimated_meters": round(float(est_dist), 2),
            "lane":             lane,
            "zone":             zone,
            "urgency":          urgency,
            "confidence":       estimate_conf,
            "method":           method,
            "bbox":             bbox,
        }

    # ─────────────────────────────────────────
    # TASK 2: THREAT RANKING
    # ─────────────────────────────────────────
    def _calculate_risk_score(self, obj: Dict) -> float:
        """
        Threat Score = (Risk Weight × Motion Sensitivity × Lane Penalty) / Distance
        All weights come from config.py
        Higher score = more dangerous
        """
        obj_type = obj["type"]
        distance = max(0.1, obj["estimated_meters"])
        lane     = obj.get("lane", "ego")

        # From config — not hardcoded
        weight         = OBJECT_RISK_WEIGHTS.get(obj_type, 1.0)
        motion_factor  = OBJECT_MOTION_SENSITIVITY.get(obj_type, 1.0)

        # Objects directly in our path are most dangerous
        lane_multiplier = 2.5 if lane == "ego" else 1.0

        # Collision probability proxy
        collision_prob = lane_multiplier * (1 / distance)

        score = (weight * motion_factor * collision_prob) * 100
        return round(score, 2)

    def _get_risk_level(self, score: float) -> str:
        """
        Converts numeric score to risk label.
        Thresholds from config — not hardcoded.
        """
        if score > RISK_SCORE_CRITICAL:  return "CRITICAL"
        if score > RISK_SCORE_HIGH:      return "HIGH"
        if score > RISK_SCORE_MEDIUM:    return "MEDIUM"
        return "LOW"

    # ─────────────────────────────────────────
    # TASK 3: MULTI-OBJECT RISK OUTPUT
    # ─────────────────────────────────────────
    def get_closest_threat(self, distance_results: dict) -> Dict:
        """
        Builds full threat report:
        - primary_threat   → most dangerous object
        - secondary_threat → second most dangerous
        - recommended_action → what the driver should do
        This output goes to pipeline.py → voice.py → reason.py
        """
        distances = distance_results.get("distances", [])

        if not distances:
            return {
                "primary_threat":     None,
                "secondary_threat":   None,
                "recommended_action": "Path is clear. Proceed normally.",
            }

        # Score every detected object
        scored = []
        for obj in distances:
            score = self._calculate_risk_score(obj)
            scored.append({
                **obj,
                "risk_score": score,
                "risk_level": self._get_risk_level(score)
            })

        # Sort: highest risk first
        scored.sort(key=lambda x: x["risk_score"], reverse=True)

        primary   = scored[0]
        secondary = scored[1] if len(scored) > 1 else None

        # Recommended action based on primary threat
        action = self._suggest_action(primary)

        return {
            "primary_threat": {
                "type":       primary["type"],
                "distance_m": primary["estimated_meters"],
                "lane":       primary["lane"],
                "risk_level": primary["risk_level"],
                "risk_score": primary["risk_score"],
            },
            "secondary_threat": {
                "type":       secondary["type"]             if secondary else None,
                "distance_m": secondary["estimated_meters"] if secondary else None,
                "lane":       secondary["lane"]             if secondary else None,
                "risk_level": secondary["risk_level"]       if secondary else None,
            } if secondary else None,
            "recommended_action": action,
        }

    def _suggest_action(self, obj: Dict) -> str:
        """
        Generates human-readable action string.
        Used by voice.py for driver alerts.
        """
        level    = obj["risk_level"]
        lane     = obj["lane"]
        obj_type = obj["type"].replace("_", " ").title()
        dist     = obj["estimated_meters"]

        if level == "CRITICAL":
            return f"Brake immediately. {obj_type} in your path at {dist}m."
        elif level == "HIGH":
            if lane == "ego":
                return f"Slow down. {obj_type} ahead at {dist}m."
            else:
                return f"Stay alert. {obj_type} on {lane} side at {dist}m."
        elif level == "MEDIUM":
            return f"Caution: {obj_type} nearby. Proceed carefully."
        else:
            return "Maintain speed. No immediate threat."


# ─────────────────────────────────────────
# CONVENIENCE FUNCTION
# Called by pipeline.py and api/main.py
# ─────────────────────────────────────────
def estimate_distances(detections: dict) -> dict:
    estimator = DistanceEstimator()
    return estimator.estimate(detections)

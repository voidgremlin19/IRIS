import sys
from pathlib import Path
# pyrefly: ignore [missing-import]
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.distance import DistanceEstimator


class TestDistanceModule:
    """Tests for perspective-aware distance estimation and threat intelligence."""

    def test_perspective_vs_bbox_ratio(self):
        """Test that perspective estimation is used when y_bottom is below horizon, and fallback is used otherwise."""
        estimator = DistanceEstimator()

        # Image height is 640. HORIZON_LINE_RATIO is 0.45. Horizon = 288.
        # Object bottom at y=400 (well below horizon)
        obj_perspective = {
            "type": "car",
            "bbox": [100, 300, 200, 400],
            "confidence": 0.8,
        }
        res_p = estimator._estimate_single(obj_perspective, 640, 640)
        assert res_p["method"] == "perspective"

        # Object bottom at y=290 (just above/at horizon threshold)
        obj_fallback = {
            "type": "car",
            "bbox": [100, 190, 200, 290],
            "confidence": 0.8,
        }
        res_f = estimator._estimate_single(obj_fallback, 640, 640)
        assert res_f["method"] == "bbox_ratio"

    def test_lane_relative_classification(self):
        """Test lane classification (left, ego, right) based on horizontal position."""
        estimator = DistanceEstimator()

        # Ego lane width ratio is 0.35. Ego lane is from 208 to 432 in 640 width image.
        
        # Center object (ego lane)
        obj_ego = {
            "type": "pedestrian",
            "bbox": [300, 400, 340, 500],
            "confidence": 0.9,
        }
        res_ego = estimator._estimate_single(obj_ego, 640, 640)
        assert res_ego["lane"] == "ego"

        # Left object
        obj_left = {
            "type": "pedestrian",
            "bbox": [50, 400, 90, 500],
            "confidence": 0.9,
        }
        res_left = estimator._estimate_single(obj_left, 640, 640)
        assert res_left["lane"] == "left"

        # Right object
        obj_right = {
            "type": "pedestrian",
            "bbox": [500, 400, 540, 500],
            "confidence": 0.9,
        }
        res_right = estimator._estimate_single(obj_right, 640, 640)
        assert res_right["lane"] == "right"

    def test_threat_ranking_and_actions(self):
        """Test risk score calculation and recommended action generation."""
        estimator = DistanceEstimator()

        # Critical threat scenario: truck close in ego lane
        critical_threat = {
            "type": "truck",
            "estimated_meters": 4.0,
            "lane": "ego",
        }
        score_crit = estimator._calculate_risk_score(critical_threat)
        level_crit = estimator._get_risk_level(score_crit)
        action_crit = estimator._suggest_action({"risk_level": level_crit, "lane": "ego", "type": "truck", "estimated_meters": 4.0})

        assert level_crit == "CRITICAL"
        assert "Brake immediately" in action_crit

        # Low threat scenario: car far on left side
        low_threat = {
            "type": "car",
            "estimated_meters": 35.0,
            "lane": "left",
        }
        score_low = estimator._calculate_risk_score(low_threat)
        level_low = estimator._get_risk_level(score_low)
        action_low = estimator._suggest_action({"risk_level": level_low, "lane": "left", "type": "car", "estimated_meters": 35.0})

        assert level_low == "LOW"
        assert "Maintain speed" in action_low

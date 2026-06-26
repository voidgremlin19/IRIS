"""
test_detect.py — Unit tests for the detection module.
"""

import sys
from pathlib import Path
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


class TestObjectDetector:
    """Tests for the ObjectDetector class."""

    def test_mock_detection_returns_objects(self):
        """Test that mock detection returns expected structure."""
        from src.detect import ObjectDetector

        detector = ObjectDetector(use_mock=True)
        # Create a dummy image
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect(dummy_image)

        assert "objects" in result
        assert "count" in result
        assert "image_size" in result
        assert result["count"] > 0

    def test_mock_detection_object_format(self):
        """Test that each detected object has required fields."""
        from src.detect import ObjectDetector

        detector = ObjectDetector(use_mock=True)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(dummy_image)

        for obj in result["objects"]:
            assert "type" in obj
            assert "bbox" in obj
            assert "confidence" in obj
            assert "bbox_normalized" in obj
            assert "area_ratio" in obj
            assert len(obj["bbox"]) == 4
            assert 0 <= obj["confidence"] <= 1

    def test_mock_detection_indian_classes(self):
        """Test that mock detections include Indian road-specific objects."""
        from src.detect import ObjectDetector

        detector = ObjectDetector(use_mock=True)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect(dummy_image)

        detected_types = {obj["type"] for obj in result["objects"]}
        # Should include some Indian road-specific types
        indian_types = {"truck", "motorcycle/bike", "pedestrian", "cow", "car"}
        assert len(detected_types & indian_types) > 0

    def test_detection_image_size(self):
        """Test that image size is correctly reported."""
        from src.detect import ObjectDetector

        detector = ObjectDetector(use_mock=True)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(dummy_image)

        assert result["image_size"]["width"] == 640
        assert result["image_size"]["height"] == 480

    def test_convenience_function(self):
        """Test the convenience function."""
        from src.detect import detect_objects

        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detect_objects(dummy_image, use_mock=True)

        assert "objects" in result
        assert result["count"] > 0


class TestSceneGraph:
    """Tests for the SceneGraphBuilder."""

    def test_scene_graph_from_detections(self):
        """Test scene graph construction."""
        from src.graph import SceneGraphBuilder

        builder = SceneGraphBuilder()
        detections = {
            "objects": [
                {
                    "type": "truck",
                    "confidence": 0.9,
                    "bbox": [200, 200, 400, 500],
                    "bbox_normalized": {"cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.4},
                    "area_ratio": 0.12,
                },
                {
                    "type": "motorcycle/bike",
                    "confidence": 0.85,
                    "bbox": [500, 300, 580, 500],
                    "bbox_normalized": {"cx": 0.8, "cy": 0.6, "w": 0.12, "h": 0.3},
                    "area_ratio": 0.036,
                },
            ],
        }

        result = builder.build(detections)

        assert "relations" in result
        assert "zones" in result
        assert "ego_view" in result
        assert result["object_count"] == 2

    def test_empty_detections(self):
        """Test scene graph with no detections."""
        from src.graph import SceneGraphBuilder

        builder = SceneGraphBuilder()
        result = builder.build({"objects": []})

        assert result["object_count"] == 0
        assert result["relations"] == []


class TestDistanceEstimation:
    """Tests for the DistanceEstimator."""

    def test_distance_estimation(self):
        """Test distance estimation output format."""
        from src.distance import DistanceEstimator

        estimator = DistanceEstimator()
        detections = {
            "objects": [
                {
                    "type": "truck",
                    "bbox": [200, 200, 400, 500],
                    "confidence": 0.9,
                },
            ],
            "image_size": {"width": 640, "height": 640},
        }

        result = estimator.estimate(detections)

        assert "distances" in result
        assert "summary" in result
        assert len(result["distances"]) == 1

        dist_obj = result["distances"][0]
        assert "estimated_meters" in dist_obj
        assert "zone" in dist_obj
        assert dist_obj["zone"] in ("near", "medium", "far")

    def test_closest_threat(self):
        """Test closest threat identification."""
        from src.distance import DistanceEstimator

        estimator = DistanceEstimator()
        distance_results = {
            "distances": [
                {"type": "truck", "estimated_meters": 5.0, "zone": "near", "lane": "ego"},
                {"type": "car", "estimated_meters": 20.0, "zone": "medium", "lane": "left"},
            ],
        }

        threat = estimator.get_closest_threat(distance_results)
        assert threat["primary_threat"]["type"] == "truck"
        assert threat["primary_threat"]["distance_m"] == 5.0


class TestReasoning:
    """Tests for the ReasoningEngine."""

    def test_rule_based_reasoning(self):
        """Test rule-based reasoning output."""
        from src.reason import ReasoningEngine

        engine = ReasoningEngine(use_llm=False)

        detections = {
            "objects": [
                {"type": "truck", "confidence": 0.9, "bbox_normalized": {"cx": 0.5, "cy": 0.5}},
                {"type": "pedestrian", "confidence": 0.8, "bbox_normalized": {"cx": 0.2, "cy": 0.6}},
            ],
        }
        scene_graph = {
            "ego_view": {
                "ahead": [{"type": "truck", "confidence": 0.9, "size_indicator": "large"}],
                "left_side": [],
                "right_side": [],
                "close": [],
                "far": [],
            },
            "relations": [],
            "clusters": [],
        }
        distances = {
            "distances": [
                {"type": "truck", "estimated_meters": 8.0, "zone": "near", "urgency": "high"},
                {"type": "pedestrian", "estimated_meters": 12.0, "zone": "medium", "urgency": "moderate"},
            ],
            "summary": {"near_objects": 1, "medium_objects": 1, "far_objects": 0},
        }

        result = engine.reason(detections, scene_graph, distances)

        assert "context" in result
        assert "decision" in result
        assert "risk" in result
        assert "alert" in result
        assert result["reasoning_method"] == "rule_based"


class TestTemporalBuffer:
    """Tests for the TemporalBuffer."""

    def test_buffer_add_and_retrieve(self):
        """Test adding entries and retrieving aggregated warning."""
        from src.temporal import TemporalBuffer

        buffer = TemporalBuffer(buffer_seconds=6.0)

        # Add a result
        buffer.add({
            "reasoning": {
                "decision": "slow_down",
                "risk": "medium",
                "alert": "Truck ahead",
                "object_summary": {"truck": 1},
            }
        })

        assert buffer.size == 1
        assert buffer.is_active

        warning = buffer.get_aggregated_warning()
        assert warning["frame_count"] == 1
        assert warning["decision"] == "slow_down"

    def test_buffer_clear(self):
        """Test clearing the buffer."""
        from src.temporal import TemporalBuffer

        buffer = TemporalBuffer()
        buffer.add({"reasoning": {"decision": "slow_down", "risk": "medium"}})
        buffer.clear()
        assert buffer.size == 0


class TestPipeline:
    """Tests for the Pipeline orchestrator."""

    def test_pipeline_mock_run(self):
        """Test full pipeline with mock detector."""
        from src.pipeline import Pipeline

        pipe = Pipeline(use_mock=True, enable_voice=False)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

        result = pipe.run(dummy_image, generate_voice=False, generate_viz=False)

        assert "detections" in result
        assert "scene_graph" in result
        assert "distances" in result
        assert "reasoning" in result
        assert "performance" in result
        assert result["detections"]["count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

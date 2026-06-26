import sys
from pathlib import Path
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from models.vlm.analyze import (
    bbox_iou,
    deduplicate_detections,
    attach_person_vehicle_relationships,
    attach_ego_proximity,
)


class TestVlmModule:
    """Tests for the helper heuristics of the VLM module."""

    def test_bbox_iou(self):
        """Test IOU calculation for bounding boxes."""
        boxA = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        boxB = {"x1": 150, "y1": 150, "x2": 250, "y2": 250}

        iou = bbox_iou(boxA, boxB)
        # Intersection = 50 * 50 = 2500
        # Union = 10000 + 10000 - 2500 = 17500
        # IOU = 2500 / 17500 = 0.142857
        assert abs(iou - 0.142857) < 1e-5

    def test_deduplicate_detections(self):
        """Test removing duplicate overlapping detections of same class."""
        yolo_data = {
            "detections": [
                {
                    "class": "car",
                    "bbox_pixels": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
                },
                # Highly overlapping second car
                {
                    "class": "car",
                    "bbox_pixels": {"x1": 105, "y1": 105, "x2": 195, "y2": 195},
                },
                # Different class, same region (should keep)
                {
                    "class": "person",
                    "bbox_pixels": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
                },
            ]
        }

        deduped = deduplicate_detections(yolo_data, iou_threshold=0.6)
        assert len(deduped["detections"]) == 2
        classes = [d["class"] for d in deduped["detections"]]
        assert "car" in classes
        assert "person" in classes

    def test_person_vehicle_relationships(self):
        """Test attaching riding relationship based on overlap."""
        yolo_data = {
            "detections": [
                {
                    "id": 1,
                    "class": "person",
                    "bbox_pixels": {"x1": 100, "y1": 100, "x2": 150, "y2": 200},
                },
                {
                    "id": 2,
                    "class": "motorcycle",
                    "bbox_pixels": {"x1": 90, "y1": 150, "x2": 160, "y2": 220},
                },
            ]
        }

        result = attach_person_vehicle_relationships(yolo_data)
        person = result["detections"][0]
        assert person.get("relationship") == "riding"
        assert person.get("vehicle_id") == 2

    def test_ego_proximity(self):
        """Test assigning ego lane proximity/relevance based on camera center."""
        yolo_data = {
            "detections": [
                # In center (high relevance)
                {
                    "class": "car",
                    "bbox_pixels": {"x1": 600, "y1": 300, "x2": 680, "y2": 400},
                },
                # Slightly offset (medium relevance)
                {
                    "class": "pedestrian",
                    "bbox_pixels": {"x1": 300, "y1": 300, "x2": 320, "y2": 400},
                },
                # Far edge (low relevance)
                {
                    "class": "dog",
                    "bbox_pixels": {"x1": 20, "y1": 300, "x2": 60, "y2": 350},
                },
            ]
        }

        result = attach_ego_proximity(yolo_data, image_width=1280)
        relevances = [d["ego_relevance"] for d in result["detections"]]
        assert relevances == ["high", "medium", "low"]

import sys
from pathlib import Path
# pyrefly: ignore [missing-import]
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from models.yolo.detect import (
    normalize_class_name,
    get_position_zone,
    get_size_category,
    generate_qa_flags,
)


class TestYoloDetectModule:
    """Tests for custom YOLO helper heuristics."""

    def test_class_normalization(self):
        """Test mapping class names to normalized Indian road equivalents."""
        assert normalize_class_name("person") == "pedestrian"
        assert normalize_class_name("traffic light") == "traffic_light"
        assert normalize_class_name("bike") == "motorcycle"
        assert normalize_class_name("car") == "car"  # unchanged
        assert normalize_class_name("rider") == "cyclist"
        assert normalize_class_name("stray dog") == "dog"
        assert normalize_class_name("stray_dog") == "dog"
        assert normalize_class_name("auto") == "auto_rickshaw"
        assert normalize_class_name("auto rickshaw") == "auto_rickshaw"
        assert normalize_class_name("stop sign") == "stop_sign"

    def test_position_zone(self):
        """Test left, center, right zone mapping based on center X coordinate."""
        # frame width = 600. Thresholds at 198 and 396.
        assert get_position_zone(100, 600) == "left"
        assert get_position_zone(300, 600) == "center"
        assert get_position_zone(500, 600) == "right"

    def test_size_category(self):
        """Test size classification (small, medium, large) based on area ratio."""
        # frame area = 10000. Thresholds at 500 (0.05) and 2000 (0.20)
        assert get_size_category(100, 10000) == "small"
        assert get_size_category(1000, 10000) == "medium"
        assert get_size_category(3000, 10000) == "large"

    def test_qa_flags(self):
        """Test generation of QA flags for edge cases and heuristics."""
        # Case 1: Edge crop flag (x1 near 0)
        flags1 = generate_qa_flags(2, 50, 100, 150, 640, 640, 0.9, "car", 0.02)
        assert "edge_crop" in flags1

        # Case 2: Low confidence flag
        flags2 = generate_qa_flags(50, 50, 150, 150, 640, 640, 0.45, "car", 0.02)
        assert "low_confidence" in flags2

        # Case 3: Unsupported class flag
        flags3 = generate_qa_flags(50, 50, 150, 150, 640, 640, 0.9, "cat", 0.02)
        assert "unsupported_class" in flags3

        # Case 4: Dashboard false positive heuristic (large box, bottom heavy, low confidence)
        # area ratio > 0.25, y2 > frame_h * 0.85, conf < 0.65
        flags4 = generate_qa_flags(100, 400, 500, 600, 640, 640, 0.55, "car", 0.3)
        assert "possible_dashboard_fp" in flags4

        # Case 5: Possible auto-rickshaw heuristic (truck class with small area)
        # cls = truck, area ratio < 0.08
        flags5 = generate_qa_flags(100, 100, 150, 150, 640, 640, 0.85, "truck", 0.02)
        assert "possible_auto_rickshaw" in flags5

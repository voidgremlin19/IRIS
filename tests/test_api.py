"""
test_api.py — API endpoint tests.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client():
    """Create test client."""
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_image_bytes():
    """Create a small sample image as bytes for testing."""
    try:
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some color to make it a valid image
        img[20:80, 20:80] = [100, 150, 200]
        _, buffer = cv2.imencode(".jpg", img)
        return buffer.tobytes()
    except ImportError:
        from PIL import Image
        import io
        img = Image.new("RGB", (100, 100), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        """Test health endpoint response format."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert "system" in data
        assert "version" in data


class TestInfoEndpoint:
    """Tests for the /info endpoint."""

    def test_info_returns_capabilities(self, client):
        """Test info endpoint returns system capabilities."""
        response = client.get("/info")
        assert response.status_code == 200

        data = response.json()
        assert "capabilities" in data
        assert "object_detection" in data["capabilities"]
        assert "supported_objects" in data


class TestAnalyzeEndpoint:
    """Tests for the /analyze endpoint."""

    def test_analyze_with_image(self, client, sample_image_bytes):
        """Test analysis with a sample image."""
        response = client.post(
            "/analyze",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "detections" in data
        assert "reasoning" in data
        assert "distances" in data

    def test_analyze_quick(self, client, sample_image_bytes):
        """Test quick analysis endpoint."""
        response = client.post(
            "/analyze/quick",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

        data = response.json()
        assert "decision" in data
        assert "risk" in data
        assert "alert" in data


class TestTemporalEndpoint:
    """Tests for the /temporal endpoint."""

    def test_temporal_status(self, client):
        """Test temporal buffer status endpoint."""
        response = client.get("/temporal")
        assert response.status_code == 200

        data = response.json()
        assert "buffer_active" in data
        assert "aggregated_warning" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

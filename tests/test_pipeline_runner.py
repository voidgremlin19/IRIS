import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
# pyrefly: ignore [missing-import]
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Adjust path context to simulate running from pipeline/
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

# pyrefly: ignore [missing-import]
from run_full import find_vlm_file, run


class TestPipelineRunnerModule:
    """Tests for the end-to-end VLM->LLM pipeline integration runner."""

    @patch("os.path.exists")
    def test_find_vlm_file(self, mock_exists):
        """Test that VLM file detection checks both normal and viz names."""
        # Case 1: normal exists
        mock_exists.side_effect = lambda p: p == "data/vlm_results/image_123.json"
        assert find_vlm_file("image_123") == "data/vlm_results/image_123.json"

        # Case 2: viz exists
        mock_exists.side_effect = lambda p: p == "data/vlm_results/image_123_viz.json"
        assert find_vlm_file("image_123") == "data/vlm_results/image_123_viz.json"

        # Case 3: neither exists
        mock_exists.side_effect = lambda p: False
        assert find_vlm_file("image_123") is None

    @patch("run_full.find_vlm_file")
    @patch("run_full.decide")
    @patch("run_full.speak_with_buffer")
    def test_run_pipeline(self, mock_speak, mock_decide, mock_find_vlm):
        """Test that run successfully finds VLM results, calls LLM decide, and triggers TTS speech."""
        mock_find_vlm.return_value = "data/vlm_results/image_123.json"
        mock_decide.return_value = {"voice_message": "Warning: auto rickshaw ahead."}

        run("/path/to/image_123.jpg")

        mock_find_vlm.assert_called_once_with("image_123")
        mock_decide.assert_called_once_with("data/vlm_results/image_123.json")
        mock_speak.assert_called_once_with("Warning: auto rickshaw ahead.")

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from models.voice.speaker import speak, speak_with_buffer, get_engine


class TestSpeakerModule:
    """Tests for the Voice Speaker offline TTS script."""

    @patch("pyttsx3.init")
    def test_speak_calls_engine(self, mock_init):
        """Test that the speak function initializes engine and calls say/runAndWait."""
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine

        # Reset global engine to force re-initialization with mock
        import models.voice.speaker
        models.voice.speaker.engine = None

        speak("Warning: Speed bump ahead")

        mock_init.assert_called_once()
        mock_engine.say.assert_called_once_with("Warning: Speed bump ahead")
        mock_engine.runAndWait.assert_called_once()

    @patch("models.voice.speaker.speak")
    @patch("time.sleep")
    def test_speak_with_buffer(self, mock_sleep, mock_speak):
        """Test speak_with_buffer delays and then calls speak."""
        speak_with_buffer("Watch out", delay=3)

        mock_sleep.assert_called_once_with(3)
        mock_speak.assert_called_once_with("Watch out")

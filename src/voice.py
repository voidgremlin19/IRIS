"""
voice.py — Voice Output Module
Converts driving decisions into spoken alerts using TTS.
Supports both gTTS (backend) and browser Speech API (frontend).
"""

import io
import base64
import logging
from typing import Optional

from src.config import TTS_LANGUAGE, TTS_SLOW, TTS_TLD

logger = logging.getLogger(__name__)


class VoiceEngine:
    """
    Text-to-Speech engine for driving alerts.
    Uses gTTS for backend speech synthesis.
    Returns audio as base64-encoded MP3 for API consumption.
    """

    def __init__(self, language: str = TTS_LANGUAGE, tld: str = TTS_TLD, slow: bool = TTS_SLOW):
        self.language = language
        self.tld = tld
        self.slow = slow
        self._gtts_available = False

        try:
            from gtts import gTTS
            self._gtts_available = True
        except ImportError:
            logger.warning("gTTS not installed. Voice output will return text only.")

    def generate_alert_audio(self, alert_text: str) -> dict:
        """
        Generate audio alert from text.
        
        Args:
            alert_text: The alert message to speak
            
        Returns:
            Dictionary with text, audio_base64, and format info
        """
        result = {
            "text": alert_text,
            "audio_base64": None,
            "format": "mp3",
            "engine": "none",
        }

        if not alert_text or alert_text.strip() == "":
            result["text"] = "All clear."
            alert_text = "All clear."

        if self._gtts_available:
            try:
                from gtts import gTTS

                tts = gTTS(text=alert_text, lang=self.language, tld=self.tld, slow=self.slow)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)

                audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
                result["audio_base64"] = audio_base64
                result["engine"] = "gtts"

                logger.info(f"Generated voice alert: {alert_text[:50]}...")

            except Exception as e:
                logger.error(f"TTS generation failed: {e}")
                result["engine"] = "error"
        else:
            # Fallback: return text only, frontend will use browser Speech API
            result["engine"] = "browser_fallback"

        return result

    def build_alert_message(self, reasoning: dict) -> str:
        """
        Build a natural-language alert message from reasoning output.
        
        Args:
            reasoning: Output from reason.py
            
        Returns:
            Natural language alert string
        """
        decision = reasoning.get("decision", "proceed_normally")
        risk = reasoning.get("risk", "none")
        alert = reasoning.get("alert", "")
        context = reasoning.get("context", "")

        # If there's already a good alert, use it
        if alert and len(alert) > 10:
            return alert

        # Build from decision
        decision_messages = {
            "emergency_brake": "Emergency! Brake immediately!",
            "emergency_caution": "Danger ahead! Extreme caution required!",
            "extreme_caution": "Caution! Multiple hazards detected. Slow down immediately.",
            "slow_down": "Slow down. Obstacle detected ahead.",
            "maintain_lane": "Stay in your lane. Vehicle overtaking.",
            "caution": "Drive carefully. Objects nearby.",
            "proceed_normally": "Road is clear. Drive safely.",
        }

        message = decision_messages.get(decision, "Stay alert.")

        # Add risk context for high-risk situations
        if risk in ("critical", "high"):
            message = f"Warning! {message}"

        return message


def generate_voice_alert(reasoning: dict) -> dict:
    """
    Convenience function to generate voice alert from reasoning.
    
    Args:
        reasoning: Output from reason.py
        
    Returns:
        Voice alert with text and optional audio
    """
    engine = VoiceEngine()
    message = engine.build_alert_message(reasoning)
    return engine.generate_alert_audio(message)

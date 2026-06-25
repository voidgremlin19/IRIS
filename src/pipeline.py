"""
pipeline.py — Pipeline Orchestrator
End-to-end processing: Image → Detection → Scene Graph → Distance → Reasoning → Voice
"""

import logging
import time
import base64
from typing import Optional, Union
from pathlib import Path
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from src.detect import ObjectDetector
from src.graph import SceneGraphBuilder
from src.distance import DistanceEstimator
from src.reason import ReasoningEngine
from src.temporal import TemporalBuffer
from src.voice import VoiceEngine
from src.viz import Visualizer

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main orchestrator for the Indian Road Intelligence System.
    Coordinates all modules in sequence:
    
    detect → graph → distance → reasoning → voice → visualization
    """

    def __init__(self, use_mock: bool = False, enable_voice: bool = True):
        """
        Initialize the pipeline with all sub-modules.
        
        Args:
            use_mock: Use mock detector (no YOLO model needed)
            enable_voice: Whether to generate voice alerts
        """
        self.detector = ObjectDetector(use_mock=use_mock)
        self.graph_builder = SceneGraphBuilder()
        self.distance_estimator = DistanceEstimator()
        self.reasoning_engine = ReasoningEngine()
        self.temporal_buffer = TemporalBuffer()
        self.voice_engine = VoiceEngine() if enable_voice else None
        self.visualizer = Visualizer()

        self.enable_voice = enable_voice
        logger.info("Pipeline initialized successfully.")

    def run(
        self,
        image: Union[np.ndarray, str, Path],
        generate_voice: bool = True,
        generate_viz: bool = True,
    ) -> dict:
        """
        Run the complete analysis pipeline on an image.
        
        Args:
            image: Input image as numpy array, file path, or Path object
            generate_voice: Whether to generate voice alert
            generate_viz: Whether to generate visualization
            
        Returns:
            Complete analysis result dictionary
        """
        start_time = time.time()

        # ─── Step 0: Load Image ──────────────────────────────────
        img_array = self._load_image(image)
        if img_array is None:
            return self._error_result("Failed to load image")

        # ─── Step 1: Object Detection ────────────────────────────
        t1 = time.time()
        detections = self.detector.detect(img_array)
        detection_time = time.time() - t1
        logger.info(f"Detection: {detections['count']} objects in {detection_time:.3f}s")

        # ─── Step 2: Scene Graph ─────────────────────────────────
        t2 = time.time()
        scene_graph = self.graph_builder.build(detections)
        graph_time = time.time() - t2
        logger.info(f"Scene graph: {len(scene_graph['relations'])} relations in {graph_time:.3f}s")

        # ─── Step 3: Distance Estimation ─────────────────────────
        t3 = time.time()
        distances = self.distance_estimator.estimate(detections)
        distance_time = time.time() - t3
        logger.info(f"Distance estimation in {distance_time:.3f}s")

        # ─── Step 4: Reasoning ───────────────────────────────────
        t4 = time.time()
        reasoning = self.reasoning_engine.reason(detections, scene_graph, distances)
        reasoning_time = time.time() - t4
        logger.info(
            f"Reasoning: decision={reasoning['decision']}, "
            f"risk={reasoning['risk']} in {reasoning_time:.3f}s"
        )

        # ─── Step 5: Voice Alert ─────────────────────────────────
        voice_alert = None
        if generate_voice and self.enable_voice:
            t5 = time.time()
            voice_alert = self.voice_engine.generate_alert_audio(
                self.voice_engine.build_alert_message(reasoning)
            )
            voice_time = time.time() - t5
            logger.info(f"Voice alert generated in {voice_time:.3f}s")

        # ─── Step 6: Visualization ───────────────────────────────
        viz_base64 = None
        if generate_viz and HAS_CV2:
            t6 = time.time()
            viz_image = self.visualizer.create_full_visualization(
                img_array, detections, distances, reasoning
            )
            # Encode visualization to base64
            _, buffer = cv2.imencode(".jpg", viz_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            viz_base64 = base64.b64encode(buffer).decode("utf-8")
            viz_time = time.time() - t6
            logger.info(f"Visualization in {viz_time:.3f}s")

        # ─── Step 7: Temporal Buffer ─────────────────────────────
        result = {
            "detections": detections,
            "scene_graph": scene_graph,
            "distances": distances,
            "reasoning": reasoning,
            "voice_alert": voice_alert,
            "visualization": viz_base64,
        }

        self.temporal_buffer.add(result)
        temporal_warning = self.temporal_buffer.get_aggregated_warning()

        # ─── Assemble Final Result ────────────────────────────────
        total_time = time.time() - start_time

        result["temporal"] = temporal_warning
        result["performance"] = {
            "total_seconds": round(total_time, 3),
            "detection_seconds": round(detection_time, 3),
            "graph_seconds": round(graph_time, 3),
            "distance_seconds": round(distance_time, 3),
            "reasoning_seconds": round(reasoning_time, 3),
        }
        result["timestamp"] = time.time()

        logger.info(f"Pipeline complete in {total_time:.3f}s")
        return result

    def _load_image(self, image: Union[np.ndarray, str, Path]) -> Optional[np.ndarray]:
        """Load image from various input types."""
        if isinstance(image, np.ndarray):
            return image

        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                logger.error(f"Image file not found: {path}")
                return None

            if HAS_CV2:
                img = cv2.imread(str(path))
                if img is None:
                    logger.error(f"Failed to read image: {path}")
                return img
            else:
                # Fallback: use PIL
                try:
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(str(path)).convert("RGB")
                    return np.array(pil_img)[:, :, ::-1]  # RGB to BGR
                except Exception as e:
                    logger.error(f"Failed to load image with PIL: {e}")
                    return None

        logger.error(f"Unsupported image type: {type(image)}")
        return None

    def _error_result(self, message: str) -> dict:
        """Return error result."""
        return {
            "error": message,
            "detections": {"objects": [], "count": 0},
            "scene_graph": {"relations": [], "zones": {}, "ego_view": {}},
            "distances": {"distances": [], "summary": {}},
            "reasoning": {
                "context": "Error",
                "decision": "stop",
                "risk": "unknown",
                "alert": message,
            },
            "voice_alert": None,
            "visualization": None,
            "temporal": None,
            "performance": {},
        }

    def run_on_bytes(
        self,
        image_bytes: bytes,
        generate_voice: bool = True,
        generate_viz: bool = True,
    ) -> dict:
        """
        Run pipeline on raw image bytes (from API upload).
        
        Args:
            image_bytes: Raw image bytes
            generate_voice: Whether to generate voice alert
            generate_viz: Whether to generate visualization
            
        Returns:
            Complete analysis result
        """
        # Decode bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        if HAS_CV2:
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            try:
                from PIL import Image as PILImage
                import io
                pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                image = np.array(pil_img)[:, :, ::-1]
            except Exception as e:
                return self._error_result(f"Failed to decode image: {e}")

        if image is None:
            return self._error_result("Failed to decode image bytes")

        return self.run(image, generate_voice, generate_viz)

"""
detect.py — Perception Layer
Object detection using YOLO for Indian road scenes.
Handles vehicles, pedestrians, animals, and road elements.
"""

import logging
from typing import Optional
import numpy as np
from pathlib import Path

from src.config import (
    YOLO_MODEL_NAME,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    INDIAN_ROAD_CLASSES,
    INDIAN_LABELS,
    IMAGE_HEIGHT,
)

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    YOLO-based object detector optimized for Indian road conditions.
    Falls back to mock detections if YOLO is unavailable.
    """

    def __init__(self, model_name: str = YOLO_MODEL_NAME, use_mock: bool = False):
        self.model_name = model_name
        self.model = None
        self.use_mock = use_mock

        if not use_mock:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_name)
                logger.info(f"YOLO model loaded: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}. Using mock detector.")
                self.use_mock = True

    def detect(
        self,
        image: np.ndarray,
        confidence: float = YOLO_CONFIDENCE_THRESHOLD,
        iou: float = YOLO_IOU_THRESHOLD,
    ) -> dict:
        """
        Run object detection on an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            confidence: Minimum confidence threshold
            iou: IoU threshold for NMS
            
        Returns:
            Dictionary with 'objects' list containing detected items
        """
        if self.use_mock:
            return self._mock_detect(image)

        return self._yolo_detect(image, confidence, iou)

    def _yolo_detect(self, image: np.ndarray, confidence: float, iou: float) -> dict:
        """Run YOLO detection on the image."""
        results = self.model(
            image,
            conf=confidence,
            iou=iou,
            verbose=False,
        )

        detections = []
        img_h, img_w = image.shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]

                # Get class name
                raw_label = result.names.get(cls_id, "unknown")

                # Map to Indian road context label
                indian_label = INDIAN_LABELS.get(raw_label, raw_label)

                # Calculate normalized center position
                cx = (bbox[0] + bbox[2]) / 2 / img_w
                cy = (bbox[1] + bbox[3]) / 2 / img_h
                box_w = (bbox[2] - bbox[0]) / img_w
                box_h = (bbox[3] - bbox[1]) / img_h

                detections.append({
                    "type": indian_label,
                    "raw_class": raw_label,
                    "class_id": cls_id,
                    "bbox": [round(b, 1) for b in bbox],
                    "bbox_normalized": {
                        "cx": round(cx, 3),
                        "cy": round(cy, 3),
                        "w": round(box_w, 3),
                        "h": round(box_h, 3),
                    },
                    "confidence": round(conf, 3),
                    "area_ratio": round(box_w * box_h, 4),
                })

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        return {
            "objects": detections,
            "count": len(detections),
            "image_size": {"width": img_w, "height": img_h},
        }

    def _mock_detect(self, image: np.ndarray) -> dict:
        """
        Generate mock detections for testing without a real model.
        Simulates typical Indian road scene.
        """
        img_h, img_w = image.shape[:2]

        mock_objects = [
            {
                "type": "truck",
                "raw_class": "truck",
                "class_id": 7,
                "bbox": [
                    round(img_w * 0.3, 1),
                    round(img_h * 0.3, 1),
                    round(img_w * 0.65, 1),
                    round(img_h * 0.75, 1),
                ],
                "bbox_normalized": {"cx": 0.475, "cy": 0.525, "w": 0.35, "h": 0.45},
                "confidence": 0.92,
                "area_ratio": 0.1575,
            },
            {
                "type": "motorcycle/bike",
                "raw_class": "motorcycle",
                "class_id": 3,
                "bbox": [
                    round(img_w * 0.7, 1),
                    round(img_h * 0.45, 1),
                    round(img_w * 0.85, 1),
                    round(img_h * 0.8, 1),
                ],
                "bbox_normalized": {"cx": 0.775, "cy": 0.625, "w": 0.15, "h": 0.35},
                "confidence": 0.87,
                "area_ratio": 0.0525,
            },
            {
                "type": "pedestrian",
                "raw_class": "person",
                "class_id": 0,
                "bbox": [
                    round(img_w * 0.1, 1),
                    round(img_h * 0.35, 1),
                    round(img_w * 0.18, 1),
                    round(img_h * 0.7, 1),
                ],
                "bbox_normalized": {"cx": 0.14, "cy": 0.525, "w": 0.08, "h": 0.35},
                "confidence": 0.81,
                "area_ratio": 0.028,
            },
            {
                "type": "car",
                "raw_class": "car",
                "class_id": 2,
                "bbox": [
                    round(img_w * 0.4, 1),
                    round(img_h * 0.5, 1),
                    round(img_w * 0.55, 1),
                    round(img_h * 0.72, 1),
                ],
                "bbox_normalized": {"cx": 0.475, "cy": 0.61, "w": 0.15, "h": 0.22},
                "confidence": 0.78,
                "area_ratio": 0.033,
            },
            {
                "type": "cow",
                "raw_class": "cow",
                "class_id": 19,
                "bbox": [
                    round(img_w * 0.05, 1),
                    round(img_h * 0.55, 1),
                    round(img_w * 0.2, 1),
                    round(img_h * 0.85, 1),
                ],
                "bbox_normalized": {"cx": 0.125, "cy": 0.7, "w": 0.15, "h": 0.30},
                "confidence": 0.73,
                "area_ratio": 0.045,
            },
        ]

        return {
            "objects": mock_objects,
            "count": len(mock_objects),
            "image_size": {"width": img_w, "height": img_h},
        }


def detect_objects(image: np.ndarray, use_mock: bool = False) -> dict:
    """
    Convenience function for one-shot detection.
    
    Args:
        image: Input image as numpy array
        use_mock: Whether to use mock detections
        
    Returns:
        Detection results dictionary
    """
    detector = ObjectDetector(use_mock=use_mock)
    return detector.detect(image)

"""
config.py — Centralized Configuration
All system-wide settings, model paths, and thresholds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = PROJECT_ROOT / "Datasets"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create output dir if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Model Configuration ─────────────────────────────────────────
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")  # nano for speed
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU", "0.45"))

# ─── Detection Classes (Indian Road Context) ─────────────────────
# COCO classes relevant to Indian roads
INDIAN_ROAD_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
    10: "fire_hydrant",
    11: "stop_sign",
    13: "bench",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    24: "backpack",
    26: "handbag",
    27: "tie",
    28: "suitcase",
}

# Friendly labels for Indian context
INDIAN_LABELS = {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle/bike",
    "bus": "bus",
    "truck": "truck",
    "traffic_light": "traffic_light",
    "stop_sign": "stop_sign",
    "cow": "cow",
    "dog": "stray_dog",
    "horse": "horse",
    "elephant": "elephant",
    "cat": "cat",
    "sheep": "sheep",
}

# Vehicle types for priority classification
HEAVY_VEHICLES = {"truck", "bus", "elephant"}
MEDIUM_VEHICLES = {"car", "motorcycle/bike", "auto_rickshaw", "bicycle"}
VULNERABLE_ROAD_USERS = {"pedestrian", "bicycle", "stray_dog", "cow", "horse", "sheep", "cat"}

# ─── Distance Estimation Parameters ──────────────────────────────
# Approximate real-world heights (in meters) for distance estimation
OBJECT_REAL_HEIGHTS = {
    "pedestrian": 1.7,
    "car": 1.5,
    "truck": 3.5,
    "bus": 3.2,
    "motorcycle/bike": 1.1,
    "bicycle": 1.0,
    "cow": 1.4,
    "stray_dog": 0.5,
    "auto_rickshaw": 1.8,
    "traffic_light": 3.0,
    "stop_sign": 2.5,
    "horse": 1.6,
    "elephant": 3.0,
}

# Distance classification thresholds (meters)
DISTANCE_NEAR_THRESHOLD = 10.0
DISTANCE_FAR_THRESHOLD = 30.0

# Camera properties
CAMERA_HEIGHT_METERS = 1.35  # standard dashcam mount height on a car
HORIZON_LINE_RATIO = 0.45  # typical horizon position
EGO_LANE_WIDTH_RATIO = 0.35  # center 35% of the frame is the ego lane

# Bbox detection confidence thresholds
CONF_LOW_THRESHOLD = 0.45
CONF_LOW_MULTIPLIER = 0.80  # treat lower-confidence objects as closer for safety
CONF_HIGH_THRESHOLD = 0.75

# Threat Risk Score Thresholds
RISK_SCORE_CRITICAL = 150.0
RISK_SCORE_HIGH = 80.0
RISK_SCORE_MEDIUM = 30.0

# Base threat weights for different objects
OBJECT_RISK_WEIGHTS = {
    "pedestrian": 10.0,
    "cow": 9.0,
    "truck": 8.0,
    "bus": 8.0,
    "motorcycle/bike": 7.0,
    "car": 6.0,
    "bicycle": 5.0,
    "stray_dog": 5.0,
    "elephant": 9.0,
    "horse": 7.0,
    "auto_rickshaw": 6.5,
    "traffic_light": 2.0,
    "stop_sign": 2.0,
    "default": 1.0,
}

# Motion sensitivity factor (e.g. how erratic the object moves)
OBJECT_MOTION_SENSITIVITY = {
    "pedestrian": 1.3,
    "stray_dog": 1.4,
    "cow": 1.2,
    "motorcycle/bike": 1.2,
    "auto_rickshaw": 1.1,
    "bicycle": 1.1,
    "car": 1.0,
    "truck": 0.8,
    "bus": 0.8,
    "elephant": 0.8,
    "horse": 1.1,
    "default": 1.0,
}

# Camera focal length estimate (pixels) — approximate for standard dashcam
CAMERA_FOCAL_LENGTH = 800.0
IMAGE_HEIGHT = 640  # default processing height

# ─── Reasoning Configuration ─────────────────────────────────────
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
USE_LLM_REASONING = os.getenv("USE_LLM", "false").lower() == "true"

# Risk level priorities
RISK_LEVELS = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "none": 0,
}

# ─── Temporal Buffer ─────────────────────────────────────────────
TEMPORAL_BUFFER_SECONDS = 6.0
TEMPORAL_MAX_FRAMES = 30

# ─── Voice / TTS ─────────────────────────────────────────────────
TTS_LANGUAGE = "en"
TTS_SLOW = False
TTS_TLD = "co.in"  # 'co.in' for Indian accent, 'com' for US, 'co.uk' for UK English

# ─── API Configuration ───────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

# ─── Visualization ───────────────────────────────────────────────
VIZ_COLORS = {
    "pedestrian": (0, 255, 0),
    "car": (255, 165, 0),
    "truck": (255, 0, 0),
    "bus": (255, 0, 0),
    "motorcycle/bike": (0, 200, 255),
    "bicycle": (0, 200, 255),
    "cow": (128, 0, 128),
    "stray_dog": (128, 0, 128),
    "traffic_light": (255, 255, 0),
    "stop_sign": (255, 255, 0),
    "auto_rickshaw": (0, 128, 255),
    "default": (200, 200, 200),
}

VIZ_FONT_SCALE = 0.5
VIZ_THICKNESS = 2
VIZ_BBOX_THICKNESS = 2

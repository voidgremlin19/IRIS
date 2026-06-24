"""
main.py — FastAPI Application
Main entry point for the Indian Road Intelligence System API.
"""

import logging
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline
from src.config import CORS_ORIGINS, API_HOST, API_PORT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("api")

#global pipeline instance 
pipeline: Pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    global pipeline
    logger.info(" Starting Indian Road Intelligence System API...")

    # Try with real YOLO model first, fall back to mock
    try:
        pipeline = Pipeline(use_mock=False, enable_voice=True)
        logger.info(" Pipeline initialized with YOLO model")
    except Exception as e:
        logger.warning(f"YOLO unavailable, using mock detector: {e}")
        pipeline = Pipeline(use_mock=True, enable_voice=True)

    yield

    logger.info(" Shutting down API...")

#fastapi application
app = FastAPI(
    title="Indian Road Intelligence System",
    description=(
        "AI-powered driving assistant for Indian road conditions. "
        "Detects objects, understands traffic context, estimates distances, "
        "and generates driving decisions with voice alerts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

#cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "system": "Indian Road Intelligence System",
        "version": "1.0.0",
        "pipeline_ready": pipeline is not None,
        "timestamp": time.time(),
    }


@app.get("/info", tags=["System"])
async def system_info():
    """Get system information and capabilities."""
    return {
        "name": "Indian Road Intelligence System",
        "version": "1.0.0",
        "description": "AI Driver Assistant for Indian Roads",
        "capabilities": [
            "object_detection",
            "scene_graph_generation",
            "distance_estimation",
            "driving_reasoning",
            "voice_alerts",
            "temporal_buffering",
            "visualization",
        ],
        "supported_objects": [
            "car", "truck", "bus", "motorcycle/bike", "bicycle",
            "pedestrian", "cow", "stray_dog", "horse", "elephant",
            "traffic_light", "stop_sign", "auto_rickshaw",
        ],
        "risk_levels": ["none", "low", "medium", "high", "critical"],
        "decisions": [
            "proceed_normally", "caution", "maintain_lane",
            "slow_down", "extreme_caution", "emergency_brake",
        ],
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze_image(
    file: UploadFile = File(...),
    generate_voice: bool = True,
    generate_viz: bool = True,
):
    """
    Analyze a road scene image.
    
    - **file**: Image file (JPEG, PNG)
    - **generate_voice**: Whether to generate voice alert audio
    - **generate_viz**: Whether to generate annotated visualization
    
    Returns complete analysis with detections, scene graph, distances,
    reasoning, voice alert, and visualization.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image/*",
        )

    try:
        # Read image bytes
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(
            f"Analyzing image: {file.filename} ({len(image_bytes)} bytes)"
        )

        # Run pipeline
        result = pipeline.run_on_bytes(
            image_bytes,
            generate_voice=generate_voice,
            generate_viz=generate_viz,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Format response
        response = {
            "success": True,
            "filename": file.filename,
            "detections": {
                "objects": result["detections"]["objects"],
                "count": result["detections"]["count"],
                "image_size": result["detections"].get("image_size"),
            },
            "scene_graph": {
                "relations": result["scene_graph"]["relations"][:20],
                "zones": result["scene_graph"]["zones"],
                "ego_view": result["scene_graph"]["ego_view"],
                "clusters": result["scene_graph"].get("clusters", []),
            },
            "distances": result["distances"],
            "reasoning": result["reasoning"],
            "voice_alert": result.get("voice_alert"),
            "visualization": result.get("visualization"),
            "temporal": result.get("temporal"),
            "performance": result.get("performance"),
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/quick", tags=["Analysis"])
async def analyze_quick(file: UploadFile = File(...)):
    """
    Quick analysis — returns only reasoning and alert (no visualization or voice).
    Optimized for speed.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        image_bytes = await file.read()
        result = pipeline.run_on_bytes(
            image_bytes,
            generate_voice=False,
            generate_viz=False,
        )

        return {
            "success": True,
            "decision": result["reasoning"]["decision"],
            "risk": result["reasoning"]["risk"],
            "alert": result["reasoning"]["alert"],
            "context": result["reasoning"]["context"],
            "object_count": result["detections"]["count"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/temporal", tags=["Analysis"])
async def get_temporal_status():
    """Get the current temporal buffer status and aggregated warning."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return {
        "buffer_active": pipeline.temporal_buffer.is_active,
        "buffer_size": pipeline.temporal_buffer.size,
        "aggregated_warning": pipeline.temporal_buffer.get_aggregated_warning(),
    }


#entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )

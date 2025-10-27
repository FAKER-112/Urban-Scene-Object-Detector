# single_api.py
import os, sys
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
# REMOVE: from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import json
import base64  # <-- ADD THIS

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils.logger import logger

app = FastAPI(title="Object Detection Single API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REMOVE: runs_directory = os.path.join(project_root, "runs")
# REMOVE: os.makedirs(runs_directory, exist_ok=True) 

class SingleDetectionResponse(BaseModel):
    filename: str
    metrics: dict
    error: Optional[str] = None
    output_image_base64: Optional[str] = None  # <-- ADD THIS FIELD

# Create a shared pipeline instance
prediction_pipeline = PredictionPipeline()

# REMOVE: app.mount("/static", ...)

@app.post("/predict", response_model=SingleDetectionResponse)
async def predict_single(
    file: UploadFile = File(...),
    conf_threshold: Optional[float] = Query(None, description="Confidence threshold (0-1)"),
    device: Optional[str] = Query(None, description="Device to run inference on, e.g., cpu or cuda")
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = file.filename or "unnamed"
    tmp_path = None
    try:
        # ... (Your logic for conf_threshold and device is fine) ...
        if conf_threshold is not None:
             try:
                 prediction_pipeline.conf_threshold = float(conf_threshold)
             except Exception:
                 raise HTTPException(status_code=400, detail="conf_threshold must be a number between 0 and 1")
        if device is not None:
             setattr(prediction_pipeline, "device", device)

        # Save uploaded file to a temporary path
        suffix = os.path.splitext(filename)[1] or ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Call the PredictionPipeline
        metrics = prediction_pipeline.predict(filename=tmp_path)

        # --- NEW: Read and encode the output image ---
        output_image_base64 = None
        output_image_path = None # For logging
        try:
            # Construct the output image path from the metrics
            if (metrics.get("output_dir") and 
                metrics.get("detection_summary") and 
                len(metrics["detection_summary"]) > 0):
                
                # --- THIS IS THE FIX ---
                # Use the original filename, not the temporary one
                output_filename = filename 
                # --- END OF FIX ---
                
                # 'metrics["output_dir"]' is relative to the CWD.
                # os.path.abspath() will correctly join CWD + relative path.
                output_dir_absolute = os.path.abspath(metrics["output_dir"])
                
                output_image_path = os.path.join(output_dir_absolute, output_filename)
                
                logger.info(f"Attempting to read output image from: {output_image_path}")

                if os.path.exists(output_image_path):
                    with open(output_image_path, "rb") as img_file:
                        img_data = img_file.read()
                        output_image_base64 = base64.b64encode(img_data).decode('utf-8')
                    logger.info(f"Successfully encoded image from: {output_image_path}")
                else:
                    logger.warning(f"Output image NOT FOUND at: {output_image_path}")
            
        except Exception as e:
            logger.error(f"Failed to read or encode output image. Path was {output_image_path}. Error: {e}")
        
        # Return metrics AND the encoded image
        return SingleDetectionResponse(
            filename=filename, 
            metrics=metrics, 
            output_image_base64=output_image_base64
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failed for {filename}: {e}")
        return SingleDetectionResponse(filename=filename, metrics={}, error=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    uvicorn.run("single_api:app", host="0.0.0.0", port=8001, reload=True)
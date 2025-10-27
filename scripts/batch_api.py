# batch_api.py
import os
import sys
import tempfile
import shutil
import base64  # <-- ADD THIS

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.pipeline.inference_pipeline import InferencePipeline
from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils.logger import logger

app = FastAPI(title="Object Detection Batch API")

# Allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileMetrics(BaseModel):
    filename: str
    total_detections: Optional[int] = None
    avg_confidence: Optional[float] = None
    output_dir: Optional[str] = None
    is_video: Optional[bool] = None
    total_time_sec: Optional[float] = None
    error: Optional[str] = None
    raw: Optional[dict] = None
    output_image_base64: Optional[str] = None  # <-- ADD THIS FIELD

class BatchResponse(BaseModel):
    batch_metrics: dict
    details: List[FileMetrics]

@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    conf_threshold: float = Query(None, description="Optional confidence threshold to override config"),
    device: str = Query(None, description="Optional device override, e.g. cpu or cuda")
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    tmp_input_dir = tempfile.mkdtemp(prefix="batch_input_")
    tmp_output_dir = tempfile.mkdtemp(prefix="batch_output_")

    try:
        saved_paths = []
        original_filenames = {}  # Map temp path to original filename
        for upload in files:
            name = upload.filename or "unnamed"
            dest = os.path.join(tmp_input_dir, name)
            try:
                with open(dest, "wb") as f:
                    f.write(await upload.read())
                saved_paths.append(dest)
                original_filenames[dest] = name  # Store original name
            except Exception as e:
                logger.error(f"Failed to save upload {name}: {e}")
        
        if not saved_paths:
            raise HTTPException(status_code=400, detail="No files could be saved for inference")

        pipeline = InferencePipeline(
            config_path="configs/pipeline_params.yaml",
            input_dir=tmp_input_dir,
            output_dir=tmp_output_dir,
        )

        if device:
            pipeline.device = device
        if conf_threshold is not None:
            pipeline.conf_threshold = float(conf_threshold)

        batch_metrics, metrics_list = pipeline.run()

        details = []
        for m in metrics_list:
            if isinstance(m, dict) and m.get("error"):
                details.append(FileMetrics(
                    filename=os.path.basename(m.get("file", "unknown")),
                    error=m.get("error"),
                    raw=m
                ))
                continue

            # This is the original filename
            filename = os.path.basename(m.get("input_source", "unknown"))
            
            # --- START: Base64 Encoding Logic ---
            output_image_base64 = None
            output_dir = m.get("output_dir") # This is relative, e.g., 'runs\detect\...'

            if output_dir:
                try:
                    # 'output_dir' is relative to the CWD. os.path.abspath resolves this.
                    output_dir_absolute = os.path.abspath(output_dir)
                    
                    # Use the original filename to find the output image
                    output_image_path = os.path.join(output_dir_absolute, filename)
                    
                    logger.info(f"Batch: Attempting to read output image from: {output_image_path}")
                    if os.path.exists(output_image_path):
                        with open(output_image_path, "rb") as img_file:
                            output_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                        logger.info(f"Batch: Successfully encoded image: {filename}")
                    else:
                        logger.warning(f"Batch: Output image NOT FOUND at: {output_image_path}")
                except Exception as e:
                    logger.error(f"Batch: Failed to read/encode image {filename}: {e}")
            # --- END: Base64 Encoding Logic ---

            details.append(FileMetrics(
                filename=filename,
                total_detections=m.get("total_detections"),
                avg_confidence=m.get("avg_confidence"),
                output_dir=m.get("output_dir"),
                is_video=m.get("is_video"),
                total_time_sec=m.get("total_time_sec"),
                raw=m,
                output_image_base64=output_image_base64  # <-- ADD THE ENCODED STRING
            ))

        return BatchResponse(batch_metrics=batch_metrics, details=details)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp input dir
        shutil.rmtree(tmp_input_dir, ignore_errors=True)
        # Note: We don't clean up tmp_output_dir as the pipeline saves to 'runs/'
        # If your pipeline *does* save to tmp_output_dir, you can clean it too
        shutil.rmtree(tmp_output_dir, ignore_errors=True)
        pass

if __name__ == "__main__":
    uvicorn.run("batch_api:app", host="0.0.0.0", port=8000, reload=True)
# Pipeline Directory

This directory contains the core orchestration pipelines for training, inference, and evaluation workflows in the Urban Scene Object Detector system.

---

## Overview

| Pipeline | Status | Description |
|----------|--------|-------------|
| `predict_pipeline.py` | Active | Single image/video inference with metrics |
| `inference_pipeline.py` | Active | Batch processing with concurrent execution |
| `train_pipeline.py` | Placeholder | End-to-end training orchestration |
| `evaluate_pipeline.py` | Active | Evaluation report generation wrapper |

---

## Detailed Documentation

### 1. `predict_pipeline.py` - Single Prediction Pipeline

Handles inference for individual images or videos with detailed metrics collection.

**Key Features:**
- Accepts both file paths and file-like objects as input
- Supports image formats (JPG, PNG, BMP, WEBP) and video formats (MP4, AVI, MOV, MKV)
- Calculates performance metrics (inference time, FPS for videos, confidence scores)
- Exports annotated results and JSON metrics to timestamped output directories
- Optional MLflow logging for experiment tracking

**Configuration (from `configs/pipeline_params.yaml`):**
| Parameter | Description |
|-----------|-------------|
| `MODEL_PATH` | Path to trained model weights |
| `CONF_THRESHOLD` | Detection confidence threshold (0-1) |
| `OUTPUT_DIR` | Base directory for prediction outputs |
| `USE_MLFLOW` | Enable MLflow logging |
| `MLFLOW_EXPERIMENT` | MLflow experiment name |

**Usage:**
```python
from src.pipeline.predict_pipeline import PredictionPipeline

pipeline = PredictionPipeline(config_path="configs/pipeline_params.yaml")
metrics = pipeline.predict(filename="path/to/image.jpg")
```

---

### 2. `inference_pipeline.py` - Batch Inference Pipeline

Orchestrates parallel inference across multiple files for high-throughput processing.

**Key Features:**
- Concurrent execution using `ThreadPoolExecutor` for improved throughput
- Aggregates detection metrics across the entire batch
- Handles individual file errors gracefully without stopping the batch
- Groups all results into a single MLflow run for experiment tracking
- Saves comprehensive batch metrics to JSON

**Configuration (from `configs/pipeline_params.yaml`):**
| Parameter | Description |
|-----------|-------------|
| `INPUT_DIR` | Directory containing input files |
| `OUTPUT_DIR` | Directory for batch outputs |
| `DEVICE` | Processing device (cpu/cuda) |
| `MAX_WORKERS` | Number of parallel threads |
| `CONF_THRESHOLD` | Detection confidence threshold |
| `MODEL_PATH` | Path to trained model weights |

**Usage:**
```python
from src.pipeline.inference_pipeline import InferencePipeline

pipeline = InferencePipeline(config_path="configs/pipeline_params.yaml")
batch_metrics, details = pipeline.run()
```

---

### 3. `train_pipeline.py` - Training Pipeline (Placeholder)

Designed to orchestrate the full training workflow from data loading to model training.

**Intended Workflow:**
1. Load raw data via `LoadDataService`
2. Clean and filter annotations via `CleanDataService`
3. Convert to YOLO format via `FeatureEngineering`
4. Train the model via `TrainModel`

**Current Status:** The pipeline logic is commented out. Training is currently handled externally (e.g., in Google Colab) due to compute requirements.

**Usage (when implemented):**
```python
from src.pipeline.train_pipeline import main
main()
```

---

### 4. `evaluate_pipeline.py` - Evaluation Pipeline

A simple wrapper that triggers the evaluation report generation.

**Key Features:**
- Calls `evaluate_existing_results()` from `src/models/evaluate_model.py`
- Generates a Markdown evaluation report from existing training outputs
- Logs metrics and artifacts to MLflow

**Usage:**
```bash
python src/pipeline/evaluate_pipeline.py
```

---

## Pipeline Architecture

```
                    ┌──────────────────────────────────┐
                    │         train_pipeline           │
                    │   (Data → Clean → FE → Train)    │
                    └──────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────┐
                    │      evaluate_pipeline           │
                    │   (Generate evaluation report)   │
                    └──────────────────────────────────┘

    ┌───────────────────────┐         ┌───────────────────────┐
    │   predict_pipeline    │         │  inference_pipeline   │
    │   (Single file)       │         │  (Batch processing)   │
    └───────────────────────┘         └───────────────────────┘
              │                                 │
              └─────────────┬───────────────────┘
                            ▼
                      YOLO Model
                    (best.pt weights)
```

---

## Configuration

All pipelines read settings from:

- `configs/pipeline_params.yaml` - Inference and batch processing settings
- `configs/config.yaml` - Data processing configurations
- `configs/model_params.yaml` - Training and evaluation parameters

---

## Integration with Scripts

The pipelines are used by the following scripts in the `scripts/` directory:

| Script | Uses Pipeline |
|--------|---------------|
| `app.py` | `PredictionPipeline`, `InferencePipeline` |
| `single_api.py` | `PredictionPipeline` |
| `batch_api.py` | `InferencePipeline` |
| `run_infer.py` | `InferencePipeline` |

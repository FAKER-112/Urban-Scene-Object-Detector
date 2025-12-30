# Scripts Directory

This directory contains entry points, interfaces, and utilities for interacting with the Urban Scene Object Detector system. Below is a comprehensive overview of each component, what it does, and how to use it.

---

## Overview

| Script | Type | Description |
|--------|------|-------------|
| `app.py` | Streamlit UI | Interactive web application for single and batch detection |
| `single_api.py` | FastAPI Server | REST API for single image/video inference |
| `batch_api.py` | FastAPI Server | REST API for batch inference on multiple files |
| `run_infer.py` | CLI Script | Command-line interface for batch inference |
| `index.html` | Web Frontend | Browser-based interface for API interaction |
| `run_train.py` | CLI Script | Placeholder for training execution |
| `run_eval.py` | CLI Script | Placeholder for evaluation execution |

---

## Detailed Documentation

### 1. `app.py` - Streamlit Web Application

A full-featured Streamlit application providing a user-friendly GUI for object detection.

**Features:**
- **Single File Detection:** Upload and analyze individual images or videos with real-time feedback.
- **Batch Processing:** Upload multiple files for parallelized inference with aggregated statistics.
- **Interactive Visualization:** Display annotated results directly within the browser.
- **Download Results:** Download processed images/videos with bounding box annotations.

**Usage:**
```bash
streamlit run scripts/app.py
```

**Default URL:** `http://localhost:8501`

---

### 2. `single_api.py` - Single Prediction REST API

A FastAPI-based REST API for processing individual image or video uploads.

**Endpoint:**
- `POST /predict` - Accepts a single file and returns detection results.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `conf_threshold` | float | Confidence threshold (0-1) for detections |
| `device` | string | Device to use (`cpu` or `cuda`) |

**Response:** Returns detection metrics and a Base64-encoded annotated image.

**Usage:**
```bash
python scripts/single_api.py
# or
uvicorn scripts.single_api:app --host 0.0.0.0 --port 8001
```

**Default URL:** `http://localhost:8001`

---

### 3. `batch_api.py` - Batch Prediction REST API

A FastAPI-based REST API for processing multiple files concurrently.

**Endpoint:**
- `POST /predict/batch` - Accepts multiple files and returns batch detection results.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `conf_threshold` | float | Confidence threshold (0-1) for detections |
| `device` | string | Device to use (`cpu` or `cuda`) |

**Response:** Returns aggregated batch metrics and per-file results with Base64-encoded images.

**Usage:**
```bash
python scripts/batch_api.py
# or
uvicorn scripts.batch_api:app --host 0.0.0.0 --port 8000
```

**Default URL:** `http://localhost:8000`

---

### 4. `run_infer.py` - Command-Line Batch Inference

A CLI wrapper around the `InferencePipeline` for executing batch inference from the terminal.

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/pipeline_params.yaml` | Path to configuration file |
| `--input-dir` | (from config) | Input directory containing images/videos |
| `--output-dir` | (from config) | Output directory for results |
| `--model-path` | (from config) | Path to model weights |
| `--device` | (from config) | Device (`cpu`, `cuda`, `mps`) |
| `--workers` | (from config) | Number of parallel workers |
| `--no-mlflow` | False | Disable MLflow logging |

**Usage:**
```bash
python scripts/run_infer.py --input-dir data/test_images --device cpu
```

---

### 5. `index.html` - Browser Frontend

A standalone HTML/JavaScript interface for interacting with the FastAPI backends.

**Features:**
- **Tab Navigation:** Switch between single and batch prediction modes.
- **File Upload:** Drag-and-drop or click to upload images.
- **Confidence Slider:** Adjust detection confidence threshold.
- **Live Preview:** Shows input image thumbnails before processing.
- **Results Display:** Renders annotated output images and detection metrics.

**Prerequisites:**
- The `single_api.py` server must be running on port `8001`.
- The `batch_api.py` server must be running on port `8000`.

**Usage:**
1. Start both API servers.
2. Open `index.html` in a web browser.

---

### 6. `run_train.py` - Training Script (Placeholder)

Currently an empty placeholder for training pipeline execution.

**Status:** Not implemented. Training is handled externally (e.g., in Google Colab).

---

### 7. `run_eval.py` - Evaluation Script (Placeholder)

Currently an empty placeholder for evaluation pipeline execution.

**Status:** Not implemented. Evaluation is handled via `src/models/evaluate_model.py`.

---

## Quick Start

### Running the Full Web Stack

1. **Start the Single API:**
   ```bash
   python scripts/single_api.py
   ```

2. **Start the Batch API:**
   ```bash
   python scripts/batch_api.py
   ```

3. **Open the Frontend:**
   Open `scripts/index.html` in your browser.

### Running the Streamlit App (Alternative)

```bash
streamlit run scripts/app.py
```

### Running CLI Inference

```bash
python scripts/run_infer.py --input-dir path/to/images --output-dir path/to/output
```

---

## Configuration

All scripts read their default settings from YAML configuration files located in the `configs/` directory:

- `configs/pipeline_params.yaml` - Inference pipeline settings
- `configs/model_params.yaml` - Model and training parameters
- `configs/config.yaml` - Data processing configurations

You can override these settings via command-line arguments or API query parameters.

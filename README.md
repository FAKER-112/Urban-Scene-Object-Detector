# Urban Scene Object Detector

A complete YOLOv8-based object detection system for urban scene analysis. This project provides end-to-end pipelines for data processing, model training, evaluation, and inference with web-based interfaces and REST APIs.

---

## Features

- **Object Detection**: Detects traffic-related objects (person, car, bus, truck, bicycle, motorcycle) in images and videos
- **Multiple Interfaces**: Streamlit web app, FastAPI REST endpoints, and CLI tools
- **Experiment Tracking**: MLflow integration for training and evaluation metrics
- **Batch Processing**: Concurrent inference for high-throughput processing
- **YOLO Format Conversion**: Automated COCO to YOLO dataset transformation

---

## Project Structure

```
project_005/
├── configs/                    # YAML configuration files
│   ├── config.yaml             # Data processing settings
│   ├── model_params.yaml       # Training/evaluation parameters
│   └── pipeline_params.yaml    # Inference pipeline settings
├── src/
│   ├── data/                   # Data loading, cleaning, feature engineering
│   ├── models/                 # Training and evaluation modules
│   ├── pipeline/               # Inference and training orchestration
│   └── utils/                  # Logging, exceptions, config parsing
├── scripts/                    # Entry points and web interfaces
│   ├── app.py                  # Streamlit web application
│   ├── single_api.py           # FastAPI for single predictions
│   ├── batch_api.py            # FastAPI for batch predictions
│   ├── run_infer.py            # CLI for batch inference
│   └── index.html              # Browser-based frontend
├── notebooks/                  # Jupyter notebooks for exploration
├── tests/                      # Unit and integration tests
├── docker/                     # Docker configuration
├── artifacts/                  # Model weights and outputs
└── data/                       # Raw and processed datasets
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd project_005
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install as editable package** (optional):
   ```bash
   pip install -e .
   ```

---

## Quick Start

### 1. Streamlit Web Application

Interactive GUI for single and batch detection:

```bash
streamlit run scripts/app.py
```

Open `http://localhost:8501` in your browser.

### 2. REST API Servers

**Single Prediction API** (port 8001):
```bash
python scripts/single_api.py
```

**Batch Prediction API** (port 8000):
```bash
python scripts/batch_api.py
```

### 3. Browser Frontend

1. Start both API servers (see above)
2. Open `scripts/index.html` in your browser

### 4. Command-Line Inference

```bash
python scripts/run_infer.py --input-dir path/to/images --output-dir path/to/output
```

---

## Configuration

Configuration files are located in `configs/`:

| File | Purpose |
|------|---------|
| `config.yaml` | Data paths, download URLs, target classes |
| `model_params.yaml` | Training epochs, image size, model paths |
| `pipeline_params.yaml` | Inference settings, confidence thresholds |

---

## Data Pipeline

The data processing workflow transforms raw COCO-format annotations into YOLO-compatible datasets:

```
Raw Data → Clean/Filter → YOLO Conversion → Train/Val Split
```

### Run Full Data Pipeline

```python
from src.data.load_data import LoadDataService
from src.data.clean_data import CleanDataService
from src.data.feature_engineering import FeatureEngineering
from pathlib import Path

# Step 1: Download data
LoadDataService(Path("configs/config.yaml")).run()

# Step 2: Filter annotations
CleanDataService(Path("configs/config.yaml")).run()

# Step 3: Convert to YOLO format
FeatureEngineering(Path("configs/config.yaml")).run()
```

---

## Training

Training is configured via `configs/model_params.yaml`:

```bash
python src/models/train_model.py
```

Training metrics and artifacts are logged to MLflow. View with:

```bash
mlflow ui
```

---

## Evaluation

Generate evaluation reports from training outputs:

```bash
python src/pipeline/evaluate_pipeline.py
```

This creates a Markdown report with metrics, confusion matrices, and PR curves.

---

## API Reference

### Single Prediction Endpoint

```
POST http://localhost:8001/predict
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | File | Image or video file |
| `conf_threshold` | float | Confidence threshold (0-1) |
| `device` | string | `cpu` or `cuda` |

### Batch Prediction Endpoint

```
POST http://localhost:8000/predict/batch
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `files` | File[] | Multiple image/video files |
| `conf_threshold` | float | Confidence threshold (0-1) |
| `device` | string | `cpu` or `cuda` |

---

## Dependencies

Key dependencies include:

- `ultralytics` - YOLOv8 implementation
- `pycocotools` - COCO dataset utilities
- `mlflow` - Experiment tracking
- `streamlit` - Web interface
- `fastapi` / `uvicorn` - REST API framework
- `opencv-python` - Image/video processing

See `requirements.txt` for full list.

---

## Testing

Run unit tests with pytest:

```bash
pytest tests/
```

---

## License

This project is licensed under the MIT License.

---

## Author

**faker_112**

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org/)
- [MLflow](https://mlflow.org/)

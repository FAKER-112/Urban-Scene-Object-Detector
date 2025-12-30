# Models Directory

This directory contains modules for model training, evaluation, and management within the Urban Scene Object Detector system.

---

## Overview

| Module | Status | Description |
|--------|--------|-------------|
| `train_model.py` | Active | YOLOv8 training with MLflow integration |
| `evaluate_model.py` | Active | Model evaluation and report generation |
| `build_model.py` | Placeholder | Model architecture definition (not implemented) |
| `save_load.py` | Placeholder | Model serialization utilities (not implemented) |

---

## Detailed Documentation

### 1. `train_model.py` - Model Training

Manages the training process for YOLOv8 object detection models with experiment tracking.

**Key Responsibilities:**
- Loads training configuration from `configs/model_params.yaml`
- Sets up MLflow experiment tracking with autologging
- Initializes and trains YOLO models from specified checkpoints
- Logs training metrics, artifacts (weights, confusion matrix), and results to MLflow

**Configuration Options:**
| Parameter | Description |
|-----------|-------------|
| `MODEL_PATH` | Path to initial model weights |
| `DATA_YAML` | Path to YOLO data configuration |
| `PROJECT_DIR` | Output directory for training artifacts |
| `RUN_NAME` | Name for the training run |
| `EPOCHS` | Number of training epochs |
| `IMGSZ` | Input image size |
| `DEVICE` | Training device (0 for GPU, cpu for CPU) |
| `RESUME` | Whether to resume from checkpoint |

**Usage:**
```bash
python src/models/train_model.py
```

---

### 2. `evaluate_model.py` - Model Evaluation

Generates detailed performance reports from existing YOLO training outputs.

**Key Responsibilities:**
- Parses `results.csv` from training runs to extract final epoch metrics
- Generates a comprehensive Markdown report (`evaluate.md`)
- Embeds visual results including confusion matrices and precision-recall curves
- Logs evaluation metrics and artifacts to MLflow

**Configuration Options:**
| Parameter | Description |
|-----------|-------------|
| `TRAIN_DIR` | Directory containing training outputs |
| `EVAL_REPORT_PATH` | Path for generated evaluation report |

**Output Report Includes:**
- Final epoch metrics (mAP, precision, recall, loss values)
- Confusion matrix visualization
- Precision-recall curve
- Validation batch sample predictions

**Usage:**
```bash
python src/models/evaluate_model.py
```

---

### 3. `build_model.py` - Model Architecture (Placeholder)

Currently an empty placeholder for custom model architecture definitions.

**Status:** Not implemented. The project uses pre-built YOLOv8 architectures from Ultralytics.

---

### 4. `save_load.py` - Model Serialization (Placeholder)

Currently an empty placeholder for model saving and loading utilities.

**Status:** Not implemented. Model serialization is handled directly by Ultralytics YOLO and MLflow.

---

## Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   train_model   │────▶│  Training Run   │────▶│ evaluate_model  │
│    .py          │     │   (artifacts)   │     │     .py         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   MLflow Logs            weights/best.pt         evaluate.md
   Parameters             results.csv             MLflow Metrics
   Autolog Metrics        confusion_matrix.png
```

---

## Configuration Files

The modules in this directory read settings from:

- `configs/model_params.yaml` - Training and evaluation parameters

Ensure this file is properly configured before running training or evaluation.

---

## MLflow Tracking

Both training and evaluation modules integrate with MLflow for experiment tracking:

- **Experiment Name (Training):** `YOLOv8_Object_Detection`
- **Experiment Name (Evaluation):** `YOLOv8_Object_Detection_Evaluation`

View logged runs using:
```bash
mlflow ui
```

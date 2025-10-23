import os
import sys
import mlflow
import torch
import re
import cv2
import yaml
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from ultralytics import YOLO
from datetime import datetime
from src.utils.config_parser import load_config
from src.utils.logger import logger
from src.utils.exception import CustomException

def evaluate_model():
    """
    Evaluates a trained YOLOv8 model on validation data and logs results to MLflow.
    Includes mAP metrics and qualitative detection samples for error analysis.
    """

    # Load evaluation configs
    config = load_config('configs/model_params.yaml')
    cfg = config.get('EvaluateModelConfig', {})

    MODEL_PATH = cfg.get('MODEL_PATH', 'artifacts/models/runs/detect/train/weights/best.pt')
    DATA_YAML = cfg.get('DATA_YAML', 'data/processed/data.yaml')
    NUM_SAMPLES = cfg.get('NUM_SAMPLES', 5)
    PROJECT_DIR = cfg.get('PROJECT_DIR', 'artifacts/models')
    DEVICE = cfg.get('DEVICE', 'cpu')
    RUN_NAME = cfg.get('RUN_NAME', 'eval_run')

    try:
        # Start MLflow tracking
        mlflow.set_experiment("YOLOv8_Object_Detection_Evaluation")
        with mlflow.start_run(run_name=f"Eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("model_path", MODEL_PATH)
            mlflow.log_param("device", DEVICE)

            # Load model
            model = YOLO(MODEL_PATH)
            model.to(DEVICE)
            logger.info(f"Loaded model from {MODEL_PATH}")

            # Evaluate on validation set
            results = model.val(data=DATA_YAML, device=DEVICE, project=PROJECT_DIR, name=RUN_NAME)

            # Log quantitative metrics
            metrics = results.results_dict
            for key, val in metrics.items():
                safe_key = re.sub(r'[^a-zA-Z0-9_\-./ ]', '_', key)
                mlflow.log_metric(safe_key, float(val))
            logger.info("Logged evaluation metrics to MLflow")

            # Save confusion matrix and precision-recall plots
            cm_path = os.path.join(PROJECT_DIR, RUN_NAME, "confusion_matrix.png")
            pr_path = os.path.join(PROJECT_DIR, RUN_NAME, "PR_curve.png")

            if os.path.exists(cm_path):
                mlflow.log_artifact(cm_path)
            if os.path.exists(pr_path):
                mlflow.log_artifact(pr_path)

            # Load class names
            with open(DATA_YAML, 'r') as f:
                data_cfg = yaml.safe_load(f)
            class_names = data_cfg.get('names', [])

            # Generate qualitative results
            val_img_dir = os.path.join(data_cfg.get('val', ''), '')
            if os.path.exists(val_img_dir):
                img_files = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png'))]
                img_samples = img_files[:NUM_SAMPLES]

                os.makedirs(f"{PROJECT_DIR}/{RUN_NAME}/samples", exist_ok=True)
                for img_path in img_samples:
                    results = model(img_path)
                    result_img = results[0].plot()
                    out_path = os.path.join(f"{PROJECT_DIR}/{RUN_NAME}/samples", os.path.basename(img_path))
                    cv2.imwrite(out_path, result_img)
                    mlflow.log_artifact(out_path)

                logger.info(f"Saved and logged {len(img_samples)} qualitative examples")

            mlflow.log_artifact(DATA_YAML)
            logger.info("Model evaluation complete")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise CustomException(e,sys)


if __name__ == "__main__":
    evaluate_model()

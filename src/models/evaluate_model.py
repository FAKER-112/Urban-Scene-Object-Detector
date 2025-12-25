"""
This module provides functions for evaluating trained YOLOv8 models, specifically
generating detailed performance reports and logging metrics to MLflow.

The module supports two main evaluation paths:
1. Active Evaluation (EvaluateModel - commented): Loads a trained model and runs
   validation on a specified dataset to compute mAP and generate sample predictions.
2. Result-based Reporting (evaluate_existing_results): Parses existing YOLO training
   outputs (results.csv, confusion matrices, etc.) to generate a comprehensive
   Markdown report (`evaluate.md`) and syncs these results to MLflow for experiment tracking.

Key features include:
- Quantitative analysis of final epoch metrics.
- Visual reporting including confusion matrices and precision-recall curves.
- Integration with MLflow for long-term tracking of model performance.
"""

import os
import sys
import mlflow
import torch
import re
import glob
import yaml
import pandas as pd
import numpy as np

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)
from ultralytics import YOLO
from datetime import datetime
from src.utils.config_parser import load_config
from src.utils.logger import logger
from src.utils.exception import CustomException

# def EvaluateModel():
#     """
#     Evaluates a trained YOLOv8 model on validation data and logs results to MLflow.
#     Includes mAP metrics and qualitative detection samples for error analysis.
#     """

#     # Load evaluation configs
#     config = load_config('configs/model_params.yaml')
#     cfg = config.get('EvaluateModelConfig', {})

#     MODEL_PATH = cfg.get('MODEL_PATH', 'artifacts/models/runs/detect/train/weights/best.pt')
#     DATA_YAML = cfg.get('DATA_YAML', 'data/processed/data.yaml')
#     NUM_SAMPLES = cfg.get('NUM_SAMPLES', 5)
#     PROJECT_DIR = cfg.get('PROJECT_DIR', 'artifacts/models')
#     DEVICE = cfg.get('DEVICE', 'cpu')
#     RUN_NAME = cfg.get('RUN_NAME', 'eval_run')

#     try:
#         # Start MLflow tracking
#         mlflow.set_experiment("YOLOv8_Object_Detection_Evaluation")
#         with mlflow.start_run(run_name=f"Eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
#             mlflow.log_param("model_path", MODEL_PATH)
#             mlflow.log_param("device", DEVICE)

#             # Load model
#             model = YOLO(MODEL_PATH)
#             model.to(DEVICE)
#             logger.info(f"Loaded model from {MODEL_PATH}")

#             # Evaluate on validation set
#             results = model.val(data=DATA_YAML, device=DEVICE, project=PROJECT_DIR, name=RUN_NAME)

#             # Log quantitative metrics
#             metrics = results.results_dict
#             for key, val in metrics.items():
#                 safe_key = re.sub(r'[^a-zA-Z0-9_\-./ ]', '_', key)
#                 mlflow.log_metric(safe_key, float(val))
#             logger.info("Logged evaluation metrics to MLflow")

#             # Save confusion matrix and precision-recall plots
#             cm_path = os.path.join(PROJECT_DIR, RUN_NAME, "confusion_matrix.png")
#             pr_path = os.path.join(PROJECT_DIR, RUN_NAME, "PR_curve.png")

#             if os.path.exists(cm_path):
#                 mlflow.log_artifact(cm_path)
#             if os.path.exists(pr_path):
#                 mlflow.log_artifact(pr_path)

#             # Load class names
#             with open(DATA_YAML, 'r') as f:
#                 data_cfg = yaml.safe_load(f)
#             class_names = data_cfg.get('names', [])

#             # Generate qualitative results
#             val_img_dir = os.path.join(data_cfg.get('val', ''), '')
#             if os.path.exists(val_img_dir):
#                 img_files = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png'))]
#                 img_samples = img_files[:NUM_SAMPLES]

#                 os.makedirs(f"{PROJECT_DIR}/{RUN_NAME}/samples", exist_ok=True)
#                 for img_path in img_samples:
#                     results = model(img_path)
#                     result_img = results[0].plot()
#                     out_path = os.path.join(f"{PROJECT_DIR}/{RUN_NAME}/samples", os.path.basename(img_path))
#                     cv2.imwrite(out_path, result_img)
#                     mlflow.log_artifact(out_path)

#                 logger.info(f"Saved and logged {len(img_samples)} qualitative examples")

#             mlflow.log_artifact(DATA_YAML)
#             logger.info("Model evaluation complete")

#     except Exception as e:
#         logger.error(f"Evaluation failed: {e}")
#         raise CustomException(e,sys)


# if __name__ == "__main__":
#     EvaluateModel()


def evaluate_existing_results():
    """
    Generate an evaluation summary (evaluate.md) using YOLO's existing output files.
    """
    try:
        # Load config
        config = load_config("configs/model_params.yaml")
        cfg = config.get("EvaluateModelConfig", {})

        TRAIN_DIR = cfg.get("TRAIN_DIR", "artifacts/models/runs/detect/train")
        EVAL_REPORT_PATH = cfg.get(
            "EVAL_REPORT_PATH", os.path.join(TRAIN_DIR, "evaluate.md")
        )

        results_csv = os.path.join(TRAIN_DIR, "results.csv")
        metrics_txt = os.path.join(TRAIN_DIR, "results.txt")
        # visuals
        confusion_matrix_path = os.path.join(TRAIN_DIR, "confusion_matrix.png")
        pr_curve_path = os.path.join(TRAIN_DIR, "PR_curve.png")
        val_images = sorted(glob.glob(os.path.join(TRAIN_DIR, "val_batch*.jpg")))

        if not os.path.exists(results_csv):
            raise FileNotFoundError(f"results.csv not found in {TRAIN_DIR}")

        # Read results.csv
        df = pd.read_csv(results_csv)

        # Extract last epoch metrics (final results)
        final_metrics = df.iloc[-1].to_dict()
        logger.info(f"Loaded YOLO evaluation metrics from {results_csv}")

        # Create Markdown report
        with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("# 🧠 Model Evaluation Report\n\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write(f"**Evaluation Source:** `{TRAIN_DIR}`\n\n")

            f.write("## 📊 Final Epoch Metrics\n")
            for key, value in final_metrics.items():
                f.write(
                    f"- **{key}**: {round(value, 4) if isinstance(value, (float, int)) else value}\n"
                )

            if os.path.exists(metrics_txt):
                f.write("\n## 📄 YOLO Summary\n")
                with open(metrics_txt, "r") as mfile:
                    f.write("```\n" + mfile.read() + "\n```\n")

            # Add visual section
            f.write("\n## 🖼️ Visual Results\n")

            if os.path.exists(confusion_matrix_path):
                f.write(f"**Confusion Matrix:**\n\n")
                f.write(f"![Confusion Matrix]({confusion_matrix_path})\n\n")

            if os.path.exists(pr_curve_path):
                f.write(f"**Precision-Recall Curve:**\n\n")
                f.write(f"![Precision-Recall Curve]({pr_curve_path})\n\n")

            if val_images:
                f.write("**Validation Sample Predictions:**\n\n")
                for img_path in val_images:
                    f.write(f"![{os.path.basename(img_path)}]({img_path})\n\n")

            # f.write("\n## Insights\n")
            # f.write("- Model performance measured using mAP@[.5:.95] and mAP@.5.\n")
            # f.write("- Use confusion matrix to identify misclassifications.\n")
            # f.write("- Check PR curve for class-wise precision/recall balance.\n")

            # f.write("\n##  Suggested Improvements\n")
            # f.write("- Increase dataset diversity or use class-balanced sampling.\n")
            # f.write("- Experiment with longer training or image augmentations.\n")
            # f.write(
            #     "- Fine-tune on small-object subsets for better pedestrian detection.\n"
            # )

        logger.info(f"Evaluation report generated at {EVAL_REPORT_PATH}")

        # Optional: log to MLflow
        mlflow.set_experiment("YOLOv8_Object_Detection_Evaluation")
        with mlflow.start_run(
            run_name=f"Eval_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            for k, v in final_metrics.items():
                if isinstance(v, (int, float)):
                    safe_key = re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", k)
                    mlflow.log_metric(safe_key, v)
            mlflow.log_artifact(EVAL_REPORT_PATH)
            if os.path.exists(confusion_matrix_path):
                mlflow.log_artifact(confusion_matrix_path)
            if os.path.exists(pr_curve_path):
                mlflow.log_artifact(pr_curve_path)

    except Exception as e:
        logger.error(f"Evaluation summary generation failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    evaluate_existing_results()

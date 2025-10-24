import os
import sys
import json
import time
import glob
import mlflow
import concurrent.futures
from datetime import datetime

# Add project root for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from typing import Dict, List, Optional
from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.config_parser import load_config


class InferencePipeline:
    """
    Orchestrates batch inference using the PredictionPipeline.
    Handles multiple inputs, MLflow logging, and device control.
    """

    def __init__(self, config_path="configs/pipeline_params.yaml", 
        
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        ):
        try:
            config = load_config(config_path).get("PredictConfig", {})

            self.input_dir = input_dir or config.get("INPUT_DIR", "data/processed/val/images")
            self.output_dir = output_dir or config.get("OUTPUT_DIR", "runs/detect_batch")
            self.device = device or config.get("DEVICE", "cpu")
            self.use_mlflow = config.get("USE_MLFLOW", True)
            self.experiment_name = config.get("MLFLOW_EXPERIMENT", "YOLOv8_BatchInference")
            self.max_workers = config.get("MAX_WORKERS", 4)
            self.conf_threshold = config.get("CONF_THRESHOLD", 0.25)
            self.model_path = model_path or  config.get("MODEL_PATH", "artifacts/models/runs/detect/train/weights/best.pt")

            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Initialized InferencePipeline with device={self.device}")

        except Exception as e:
            raise CustomException(f"Failed to initialize InferencePipeline: {e}")

    def _run_single_inference(self, pipeline: PredictionPipeline, file_path: str):
        """Run single file prediction and return results or errors."""
        try:
            return pipeline.predict(filename=file_path)
        except Exception as e:
            logger.error(f"Inference failed for {file_path}: {e}")
            return {"file": file_path, "error": str(e)}

    def run(self):
        """Run inference across all files in input_dir."""
        try:
            start_time = time.time()

            # Collect image/video files
            input_files = glob.glob(os.path.join(self.input_dir, "*"))
            input_files = [f for f in input_files if os.path.isfile(f)]
            if not input_files:
                logger.warning("No files found for inference.")
                return

            logger.info(f"Found {len(input_files)} files for inference.")

            # Prepare model pipeline
            prediction_pipeline = PredictionPipeline()
            metrics_list = []

            # Enable MLflow
            if self.use_mlflow:
                mlflow.set_experiment(self.experiment_name)
                mlflow_run = mlflow.start_run(run_name=f"BatchInference_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                mlflow.log_param("device", self.device)
                mlflow.log_param("model_path", self.model_path)
                mlflow.log_param("conf_threshold", self.conf_threshold)

            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._run_single_inference, prediction_pipeline, f) for f in input_files]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    metrics_list.append(result)

            # Aggregate metrics
            total_detections = sum(m.get("total_detections", 0) for m in metrics_list if "error" not in m)
            avg_confidences = [m.get("avg_confidence", 0) for m in metrics_list if "error" not in m]
            avg_conf = sum(avg_confidences) / len(avg_confidences) if avg_confidences else 0

            batch_metrics = {
                "total_files": len(input_files),
                "successful_inferences": len([m for m in metrics_list if "error" not in m]),
                "failed_inferences": len([m for m in metrics_list if "error" in m]),
                "total_detections": total_detections,
                "avg_confidence": round(avg_conf, 3),
                "total_runtime_sec": round(time.time() - start_time, 3),
                "device": self.device,
            }

            # Save results
            metrics_path = os.path.join(self.output_dir, "batch_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({"batch_metrics": batch_metrics, "details": metrics_list}, f, indent=4)

            # Log batch metrics to MLflow
            if self.use_mlflow:
                mlflow.log_metrics({
                    "batch_total_detections": total_detections,
                    "batch_avg_confidence": avg_conf,
                    "batch_runtime_sec": batch_metrics["total_runtime_sec"]
                })
                mlflow.log_artifact(metrics_path)
                mlflow.end_run()
                logger.info("Batch inference metrics logged to MLflow.")

            logger.info(
                f"✅ Batch inference complete: {batch_metrics['successful_inferences']}/{batch_metrics['total_files']} "
                f"files processed | Avg Conf: {batch_metrics['avg_confidence']:.2f} | "
                f"Runtime: {batch_metrics['total_runtime_sec']}s"
            )

            return batch_metrics

        except Exception as e:
            raise CustomException(e, sys)
        
        finally:
            if self.use_mlflow and mlflow.active_run():
                mlflow.end_run()
            # Close model/pipeline resources


if __name__ == "__main__":
    pipeline = InferencePipeline(config_path="configs/pipeline_params.yaml")
    results = pipeline.run()
    print(json.dumps(results, indent=4))

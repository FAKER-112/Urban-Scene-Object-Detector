import os, sys
import time
import json
import cv2
import argparse
import supervision as sv
import mlflow
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from ultralytics import YOLO
from datetime import datetime
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.config_parser import load_config


class PredictionPipeline:
    def __init__(self, config_path="configs/pipeline_params.yaml"):
        try:
            config = load_config(config_path).get("PredictConfig", {})
            self.model_path = config.get("MODEL_PATH", "artifacts/models/runs/detect/train/weights/best.pt")
            self.conf_threshold = config.get("CONF_THRESHOLD", 0.25)
            self.output_dir = config.get("OUTPUT_DIR", "runs/detect")
            self.use_mlflow = config.get("USE_MLFLOW", False)
            self.experiment_name = config.get("MLFLOW_EXPERIMENT", "YOLOv8_Inference")
        except Exception as e:
            raise CustomException(f"Config loading failed: {e}")

    def predict(self, filename: str):
        try:
            logger.info(f"Starting prediction for: {filename}")

            # Initialize output folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_output_dir = os.path.join(self.output_dir, f"predict_{timestamp}")
            os.makedirs(run_output_dir, exist_ok=True)

            # Load model
            model = YOLO(self.model_path)

            # Handle file-like input
            temp_path = None
            if not isinstance(filename, str):
                temp_path = "temp_input"
                with open(temp_path, "wb") as f:
                    f.write(filename.read())
                filename = temp_path

            # Detect if input is video or image
            ext = os.path.splitext(filename)[1].lower()
            is_video = ext in [".mp4", ".avi", ".mov", ".mkv"]

            # Start timing
            start_time = time.time()

            # Run YOLO prediction
            results = model.predict(
                source=filename,
                save=True,
                project=run_output_dir,
                conf=self.conf_threshold,
                verbose=False,
                imgsz=960
            )

            # Stop timing
            total_time = time.time() - start_time

            # Collect metrics
            detection_summary, total_detections, total_conf = [], 0, 0.0

            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    confs = boxes.conf.cpu().numpy().tolist()
                    total_conf += sum(confs)
                    total_detections += len(confs)
                    detection_summary.append({
                        "image": os.path.basename(result.path),
                        "num_detections": len(confs),
                        "avg_confidence": sum(confs) / len(confs)
                    })

            avg_confidence = total_conf / total_detections if total_detections > 0 else 0

            # Calculate FPS for video
            fps = None
            if is_video:
                cap = cv2.VideoCapture(filename)
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = num_frames / total_time if total_time > 0 else 0
                cap.release()

            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

            # Build metrics dictionary
            metrics = {
                "model_path": self.model_path,
                "input_source": filename,
                "is_video": is_video,
                "output_dir": os.path.relpath(str(results[0].save_dir), start=os.getcwd()),
                "total_time_sec": round(total_time, 3),
                "fps": round(fps, 2) if fps else None,
                "total_detections": total_detections,
                "avg_confidence": round(avg_confidence, 3),
                "detection_summary": detection_summary
            }

            # Save metrics as JSON
            metrics_path = os.path.join(run_output_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            # Log to MLflow (optional)
            if self.use_mlflow:
                mlflow.set_experiment(self.experiment_name)
                with mlflow.start_run(run_name=f"Inference_{timestamp}"):
                    mlflow.log_param("model_path", self.model_path)
                    mlflow.log_param("conf_threshold", self.conf_threshold)
                    mlflow.log_metric("total_detections", total_detections)
                    mlflow.log_metric("avg_confidence", avg_confidence)
                    mlflow.log_metric("inference_time", total_time)
                    if fps:
                        mlflow.log_metric("fps", fps)
                    mlflow.log_artifact(metrics_path)
                    logger.info("Inference logged to MLflow.")

            logger.info(f"✅ Inference complete in {metrics['total_time_sec']}s | "
                        f"Detections: {total_detections} | Avg Conf: {metrics['avg_confidence']:.2f}")

            return metrics

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise CustomException(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Prediction Pipeline")
    parser.add_argument("--input", required=True, help="Path to image or video")
    parser.add_argument("--config", default="configs/pipeline_params.yaml", help="Path to config file")
    args = parser.parse_args()

    pipeline = PredictionPipeline(config_path=args.config)
    metrics = pipeline.predict(filename=args.input)
    print(json.dumps(metrics, indent=4))

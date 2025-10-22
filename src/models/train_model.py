import os
import sys
import mlflow
import mlflow.pytorch
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from ultralytics import YOLO
from datetime import datetime
from src.utils.config_parser import load_config
from src.utils.exception import CustomException
from src.utils.logger import logger


def main():
    try:
        # Load config
        config = load_config(config_path="configs/model_params.yaml")
        cfg = config.get("TrainModelConfig", {})

        MODEL_PATH = cfg.get("MODEL_PATH", "artifacts/models/runs/detect/train/weights/last.pt")
        DATA_YAML = cfg.get("DATA_YAML", "data/processed/data.yaml")
        PROJECT_DIR = cfg.get("PROJECT_DIR", "artifacts/models")
        RUN_NAME = cfg.get("RUN_NAME", "train")
        epochs = cfg.get("EPOCHS", 101)
        imgsz = cfg.get("IMGSZ", 640)
        device = cfg.get("DEVICE", 0)
        resume = cfg.get("RESUME", True)

        # Set MLflow experiment
        mlflow.set_experiment("YOLOv8_Object_Detection")

        with mlflow.start_run(run_name=f"YOLOv8_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters (only inside the run context)
            mlflow.log_params({
                "epochs": epochs,
                "imgsz": imgsz,
                "device": device,
                "resume": resume,
                "model_path": MODEL_PATH,
                "data_yaml": DATA_YAML
            })

            # Load and train model
            model = YOLO(MODEL_PATH)
            results = model.train(
                data=DATA_YAML,
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                project=PROJECT_DIR,
                name=RUN_NAME,
                resume=resume,
            )

            # Log metrics safely
            if hasattr(results, "results_dict"):
                for key, value in results.results_dict.items():
                    mlflow.log_metric(key, float(value))

            # Log model if available
            trained_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
            if os.path.exists(trained_weights):
                mlflow.pytorch.log_model(model, artifact_path="model")
                mlflow.log_artifact(trained_weights)

            # Log artifacts for traceability
            for artifact_file in [
                DATA_YAML,
                os.path.join(PROJECT_DIR, RUN_NAME, "results.csv"),
                os.path.join(PROJECT_DIR, RUN_NAME, "confusion_matrix.png"),
            ]:
                if os.path.exists(artifact_file):
                    mlflow.log_artifact(artifact_file)

            logger.info("Training complete. All artifacts logged to MLflow.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise CustomException(e,sys)


if __name__ == "__main__":
    main()

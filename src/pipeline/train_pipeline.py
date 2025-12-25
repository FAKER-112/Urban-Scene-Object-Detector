"""
This module defines the end-to-end training pipeline for the Urban Scene Object Detector.

The pipeline is designed to orchestrate the following stages:
1. Data Loading: Downloading raw data and labels.
2. Data Cleaning: Filtering and preprocessing annotations.
3. Feature Engineering: Converting data to YOLO format and splitting datasets.
4. Model Training: Initializing and training the YOLOv8 model.

Note: Currently, the active pipeline logic is commented out, and the `main` function
serves as a placeholder, as training is being handled externally (e.g., in Colab).
"""

# import os
# import sys
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)
# from pathlib import Path
# from src.data.load_data import LoadDataService
# from src.data.clean_data import CleanDataService
# from src.data.feature_engineering import FeatureEngineering
# from src.models.train_model import TrainModel
# from src.utils.exception import CustomException
from src.utils.logger import logger


# def main():
#     try:
#         logger.info("Taining Pipeline started")
#         loader = LoadDataService(Path("configs/config.yaml"))
#         loader.run()
#         logger.info("Data loaded")
#         cleaner = CleanDataService(Path("configs/config.yaml"))
#         cleaner.run()
#         logger.info("Data cleaned")
#         fe = FeatureEngineering(Path("configs/config.yaml"))
#         fe.run()
#         logger.info("Feature engineering completed")
#         TrainModel()
#         logger.info("Training completed")
#     except Exception as e:
#         raise CustomException(e,sys)
def main():
    logger.info("Training skipped — handled externally in Colab environment.")


if __name__ == "__main__":
    main()

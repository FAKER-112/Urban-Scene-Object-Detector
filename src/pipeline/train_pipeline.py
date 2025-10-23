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


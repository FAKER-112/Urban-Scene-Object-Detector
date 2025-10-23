import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.models.evaluate_model import evaluate_existing_results
from src.utils.exception import CustomException
from src.utils.logger import logger

def main():
    try:
        logger.info("Evaluation Pipeline started")
        evaluate_existing_results()
        logger.info("Evaluation completed")
    except Exception as e:
        raise CustomException(e,sys)


if __name__ == "__main__":
    main()
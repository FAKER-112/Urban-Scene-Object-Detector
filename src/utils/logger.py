import os
import logging
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(os.getcwd(), LOG_DIR, LOG_FILE_NAME)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename=LOG_FILE_PATH
)
logger = logging.getLogger(__name__)    
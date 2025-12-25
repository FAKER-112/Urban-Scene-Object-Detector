"""
This module provides the LoadDataService class, which handles the acquisition
and extraction of project data.

Key Responsibilities:
- Loads configuration parameters for data loading.
- Downloads image datasets and label files from Google Drive using utility functions.
- Extracts zip-compressed images into the raw data directory.
- Manages the storage paths for both raw and processed data components.

The service uses a logger to track the progress of downloading and extraction,
providing feedback on success or failure of various stages.
"""

import os
import zipfile
import sys

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)
from pathlib import Path
from src.utils.config_parser import load_config
from src.utils.utils import download_from_gdrive
from src.utils.exception import CustomException
from src.utils.logger import logger


class LoadDataService:
    """Service to load data by downloading and extracting images and labels."""

    def __init__(self, config_yaml_path: Path):

        try:
            self.config = load_config(config_yaml_path)
            self.logger = logger

            paths = self.config.get("paths", {})
            self.raw_dir = Path(paths.get("data_raw", "data/raw"))
            self.processed_dir = Path(paths.get("data_processed", "data/processed"))

            load_cfg = self.config.get("LoadDataConfig", {})
            self.image_url: str = load_cfg.get("images_url")
            self.label_url: str = load_cfg.get("label_url")
            self.image_zip_name: str = load_cfg.get("image_zip_name", "images.zip")
            self.label_filename: str = load_cfg.get("label_filename", "labels.json")
            self.unzip_dir: str = load_cfg.get("unzip_dirname", "unzipped_images")

        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        """Executes the data loading process including downloading and extracting images and labels."""

        try:
            self.logger.info("starting data loading process ...")
            image_zip = self._download_images()
            self._extract_images(image_zip)
            self._download_labels()
            self.logger.info("data loading process completed successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def _download_images(self):
        """Downloads image zip file from Google Drive."""
        try:
            os.makedirs(self.raw_dir, exist_ok=True)
            image_zip_path = self.raw_dir / self.image_zip_name

            self.logger.info(
                f"Downloading images from {self.image_url} to {image_zip_path}"
            )
            download_from_gdrive(self.image_url, str(image_zip_path))
            self.logger.info(f"Downloaded images to {image_zip_path}")
            return image_zip_path
        except Exception as e:
            raise CustomException(e, sys)

    def _extract_images(self, zip_path: Path):
        """Extracts images from the downloaded zip file."""

        try:
            target_dir = self.raw_dir / self.unzip_dir
            os.makedirs(target_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            self.logger.info(f"Extracted images to {target_dir}")
        except Exception as e:
            raise CustomException(e, sys)

    def _download_labels(self):
        """Downloads label JSON file from Google Drive."""
        try:
            os.makedirs(self.raw_dir, exist_ok=True)
            label_file_path = self.raw_dir / self.label_filename

            self.logger.info(
                f"Downloading label json from {self.label_url} to {label_file_path}"
            )
            download_from_gdrive(self.label_url, str(label_file_path))
            self.logger.info(f"Downloaded label json to {label_file_path}")
            return label_file_path
        except Exception as e:
            CustomException(e, sys)


if __name__ == "__main__":
    loader = LoadDataService(Path("configs/config.yaml"))
    loader.run()

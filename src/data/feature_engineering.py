"""
This module provides the FeatureEngineering class, which prepares the dataset
for model training by converting COCO-formatted data into the YOLO format.

The script performs:
1. Dataset Splitting: Randomly partitions the dataset into training and validation sets
   based on a configurable ratio.
2. Format Conversion: Transforms COCO bounding box coordinates (x, y, width, height)
   into YOLO-compatible normalized center-based coordinates (xc, yc, w, h).
3. Workspace Organization: Structures the processed images and corresponding YOLO
   label text files into appropriate subdirectories (train/val).
4. YAML Configuration: Generates a YOLO `data.yaml` file containing paths to
   the splits and the category names for training.

The service relies on PyTorch's CocoDetection for initial data access and logs
the transformation progress.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import random_split
from torchvision.datasets import CocoDetection
from torchvision import transforms

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.config_parser import load_config
from src.utils.exception import CustomException
from src.utils.logger import logger


class FeatureEngineering:
    """Convert COCO-format dataset into YOLO-compatible train/test structure."""

    def __init__(self, config_yaml_path: Path):
        try:
            self.config = load_config(config_yaml_path)
            self.logger = logger

            cfg = self.config.get("FeatureEngConfig", {})
            self.base_dir = Path(cfg.get("base_dir", "data/processed"))
            self.image_dir = Path(
                cfg.get("image_dir", "data/raw/urban_scene_images/images")
            )
            self.ann_dir = Path(
                cfg.get(
                    "ann_dir", "data/raw/urban_scene_images/annotation_minimal.json"
                )
            )

            clean_cfg = self.config.get("CleanDataConfig", {})
            self.target_classes = clean_cfg.get(
                "target_classes",
                ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
            )

            self.train_ratio = cfg.get("train_ratio", 0.8)

        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        try:
            self.logger.info("Starting feature engineering pipeline...")
            train_dataset, test_dataset = self._split_dataset()
            self._save_splits(train_dataset, test_dataset)
            self._create_yaml()
            self.logger.info("Feature engineering completed successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def _split_dataset(self):
        """Split the full dataset into train and test subsets."""
        transform = transforms.ToTensor()
        full_dataset = CocoDetection(
            root=self.image_dir, annFile=self.ann_dir, transform=transform
        )
        total_size = len(full_dataset)
        train_size = int(self.train_ratio * total_size)
        test_size = total_size - train_size
        self.logger.info(
            f"Dataset size: {total_size} | Train: {train_size} | Test: {test_size}"
        )
        return random_split(full_dataset, [train_size, test_size])

    def _save_splits(self, train_dataset, test_dataset):
        """Copy images and generate YOLO txt labels."""
        try:
            with open(self.ann_dir, "r") as f:
                data = json.load(f)

            id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
            id_to_size = {
                img["id"]: (img["width"], img["height"]) for img in data["images"]
            }

            img_to_boxes = {}
            for ann in data["annotations"]:
                img_id = ann["image_id"]
                bbox = ann["bbox"]
                cat_id = ann["category_id"]
                img_to_boxes.setdefault(img_id, []).append((bbox, cat_id))

            def coco_to_yolo(x, y, w, h, img_w, img_h):
                xc = (x + w / 2) / img_w
                yc = (y + h / 2) / img_h
                return xc, yc, w / img_w, h / img_h

            for split_name, dataset in zip(
                ["train", "val"], [train_dataset, test_dataset]
            ):
                img_out = self.base_dir / split_name / "images"
                lbl_out = self.base_dir / split_name / "labels"
                img_out.mkdir(parents=True, exist_ok=True)
                lbl_out.mkdir(parents=True, exist_ok=True)

                self.logger.info(f"Processing {split_name} set...")
                for i in tqdm(range(len(dataset)), desc=f"{split_name}"):
                    _, target = dataset[i]
                    img_id = target[0]["image_id"]
                    filename = id_to_file[img_id]

                    src_path = self.image_dir / filename
                    dst_path = img_out / filename
                    shutil.copy(src_path, dst_path)

                    img_w, img_h = id_to_size[img_id]
                    boxes = img_to_boxes.get(img_id, [])
                    txt_path = lbl_out / f"{Path(filename).stem}.txt"

                    with open(txt_path, "w") as f:
                        for bbox, cat_id in boxes:
                            x, y, w, h = bbox
                            xc, yc, w, h = coco_to_yolo(x, y, w, h, img_w, img_h)
                            f.write(f"{cat_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                self.logger.info(
                    f"Saved {split_name} images and labels to {img_out.parent}"
                )

        except Exception as e:
            raise CustomException(e, sys)

    def _create_yaml(self):
        """Create YOLO data configuration file."""
        try:
            yaml_path = self.base_dir / "data.yaml"
            yaml_content = {
                "train": str(self.base_dir / "train/images"),
                "val": str(self.base_dir / "val/images"),
                "nc": len(self.target_classes),
                "names": self.target_classes,
            }

            import yaml

            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(yaml_path, "w") as f:
                yaml.safe_dump(yaml_content, f, sort_keys=False)

            self.logger.info(f"Dataset YAML created: {yaml_path}")
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    FeatureEngineering(Path("configs/config.yaml")).run()

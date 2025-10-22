import os   
import sys
import json
import argparse
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from pathlib import Path
from pycocotools.coco import COCO
from src.utils.config_parser import load_config
from src.utils.exception import CustomException
from src.utils.logger import logger


class CleanDataService:
    """Filters COCO annotations to traffic-related classes."""

    def __init__(self, config_yaml_path: Path):
        try:
            self.config = load_config(config_yaml_path)
            self.logger = logger

            cfg = self.config.get("CleanDataConfig", {})
            self.src_path = Path(cfg.get("src_ann_dir", "data/raw/annotations.json"))
            self.dest_path = Path(cfg.get("dest_ann_dir", "data/processed/annotation_filtered..json"))
            self.min_path = Path(cfg.get("min_ann_dir", "data/raw/urban_scene_images/annotation_minimal.json"))
            self.target_classes =  cfg.get("target_classes", [])

        except Exception as e:
            raise CustomException(e, sys)

    def run(self) -> None:
        """Main execution entry."""
        try:
            self.logger.info("Starting data cleaning process...")
            self._filter_annotations()
            self._minimal_json()
            self.logger.info("Data cleaning process completed successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def _filter_annotations(self) -> None:
        """Filter COCO annotations to target object classes."""
        try:
            if not self.src_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {self.src_path}")

            coco = COCO(str(self.src_path))
            target_cat_ids = coco.getCatIds(catNms=self.target_classes)
            target_ann_ids = coco.getAnnIds(catIds=target_cat_ids)
            target_anns = coco.loadAnns(target_ann_ids)
            target_img_ids = list({ann["image_id"] for ann in target_anns})

            filtered = {
                "info": coco.dataset.get("info", {}),
                "licenses": coco.dataset.get("licenses", []),
                "images": [img for img in coco.dataset["images"] if img["id"] in target_img_ids],
                "annotations": [ann for ann in coco.dataset["annotations"] if ann["category_id"] in target_cat_ids],
                "categories": [cat for cat in coco.dataset["categories"] if cat["id"] in target_cat_ids],
            }

            # Reindex category IDs
            old_to_new = {old_id: i for i, old_id in enumerate(target_cat_ids)}
            for ann in filtered["annotations"]:
                ann["category_id"] = old_to_new[ann["category_id"]]
            for cat in filtered["categories"]:
                cat["id"] = old_to_new[cat["id"]]

            # Ensure parent directories exist
            self.dest_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.dest_path, "w") as f:
                json.dump(filtered, f, indent=2)

            self.logger.info(f"Filtered dataset saved to {self.dest_path}")
            self.logger.info(
                f"Images: {len(filtered['images'])}, Annotations: {len(filtered['annotations'])}, "
                f"Categories: {len(filtered['categories'])}"
            )

        except Exception as e:
            raise CustomException(e, sys)
    def _minimal_json(self)-> None:
        """Create a minimal JSON with only image IDs and file names."""
        try:
        
            # Load file
            with open(self.dest_path, "r") as f:
                data = json.load(f)

            # Keep only required keys
            cleaned = {
                "images": data["images"],
                "annotations": [
                    {k: ann[k] for k in ["bbox", "category_id", "image_id", "id"]}
                    for ann in data["annotations"]
                ],
                "categories": data["categories"]
            }

            # Save cleaned version
            with open(self.min_path, "w") as f:
                json.dump(cleaned, f)

            self.logger.info(f"Saved cleaned annotation file: {self.min_path}")
            self.logger.info(f"Images: {len(cleaned['images'])}, Annotations: {len(cleaned['annotations'])}")

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean COCO annotations for urban scene detection.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cleaner = CleanDataService(Path(args.config))
    cleaner.run()

import os
import sys
import json
import time
import glob
import mlflow
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import traceback

# Add project root for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.pipeline.inference_pipeline import InferencePipeline
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.config_parser import load_config


# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS


@dataclass
class InferenceResult:
    """Structured result for a single inference."""
    file_path: str
    success: bool
    total_detections: int = 0
    avg_confidence: float = 0.0
    inference_time_sec: float = 0.0
    detections_by_class: Optional[Dict[str, int]] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BatchMetrics:
    """Aggregated metrics for the entire batch."""
    total_files: int
    successful_inferences: int
    failed_inferences: int
    total_detections: int
    avg_confidence: float
    avg_inference_time_sec: float
    total_runtime_sec: float
    device: str
    timestamp: str
    detections_by_class: Dict[str, int]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch inference with YOLOv8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_params.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory containing images/videos (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (overrides config)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model weights (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for inference (overrides config)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (overrides config)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for batch inference."""
    args = parse_args()
    
    try:
        logger.info("Starting batch inference pipeline...")
        
        # Initialize pipeline with config and CLI overrides
        pipeline = InferencePipeline(
            config_path=args.config,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            device=args.device,
            model_path=args.model_path,
        )
        
        # Override MLflow setting if specified
        if args.no_mlflow:
            pipeline.use_mlflow = False
            logger.info("MLflow logging disabled via command line argument")
        
        # Override workers if specified
        if args.workers:
            pipeline.max_workers = args.workers
            logger.info(f"Max workers set to {args.workers}")
        
        # Run batch inference
        logger.info(f"Running inference on files in: {pipeline.input_dir}")
        logger.info(f"Device: {pipeline.device} | Workers: {pipeline.max_workers}")
        
        batch_metrics = pipeline.run()
        
        # Display summary
        if batch_metrics:
            print("\n" + "="*70)
            print("BATCH INFERENCE SUMMARY")
            print("="*70)
            print(f"Total Files:          {batch_metrics['total_files']}")
            print(f"Successful:           {batch_metrics['successful_inferences']}")
            print(f"Failed:               {batch_metrics['failed_inferences']}")
            print(f"Total Detections:     {batch_metrics['total_detections']}")
            print(f"Average Confidence:   {batch_metrics['avg_confidence']:.3f}")
            print(f"Total Runtime:        {batch_metrics['total_runtime_sec']:.2f}s")
            print(f"Device:               {batch_metrics['device']}")
            print("="*70)
            print(f"\n✓ Results saved to: {pipeline.output_dir}")
            print(f"✓ Metrics file: {os.path.join(pipeline.output_dir, 'batch_metrics.json')}")
        
        # Exit with appropriate code
        exit_code = 0 if batch_metrics and batch_metrics['failed_inferences'] == 0 else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠ Inference interrupted by user")
        sys.exit(130)
        
    except CustomException as e:
        logger.error(f"\n✗ Custom exception occurred: {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n✗ Unexpected error during inference: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
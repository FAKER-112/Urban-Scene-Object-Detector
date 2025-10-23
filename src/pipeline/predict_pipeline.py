import os
import time
import cv2
import json
from ultralytics import YOLO


def run_yolo_prediction(model_path, input_source, output_dir="runs/detect", conf_threshold=0.25):
    """
    Runs object detection on an image or video using YOLOv8 and logs key performance metrics.

    Args:
        model_path (str): Path to YOLO model (.pt file).
        input_source (str or file): Image/video path or file-like object.
        output_dir (str): Directory to save results.
        conf_threshold (float): Confidence threshold for detections.

    Returns:
        dict: Dictionary containing performance metrics and detection summary.
    """

    # Load YOLO model
    model = YOLO(model_path)

    # Handle file-like input
    temp_path = None
    if not isinstance(input_source, str):
        temp_path = "temp_input"
        with open(temp_path, "wb") as f:
            f.write(input_source.read())
        input_source = temp_path

    # Detect if input is video or image
    ext = os.path.splitext(input_source)[1].lower()
    is_video = ext in [".mp4", ".avi", ".mov", ".mkv"]

    # Start timing
    start_time = time.time()

    # Run prediction
    results = model.predict(
        source=input_source,
        save=True,
        project=output_dir,
        conf=conf_threshold,
        verbose=False
    )

    # Stop timing
    end_time = time.time()
    total_time = end_time - start_time

    # Extract metrics
    detection_summary = []
    total_detections = 0
    avg_confidence = 0

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy().tolist()
            avg_confidence += sum(confs)
            total_detections += len(confs)
            detection_summary.append({
                "image": os.path.basename(result.path),
                "num_detections": len(confs),
                "avg_confidence": sum(confs) / len(confs)
            })

    # Compute overall averages
    avg_confidence = avg_confidence / total_detections if total_detections > 0 else 0

    # Calculate FPS for video
    fps = None
    if is_video:
        cap = cv2.VideoCapture(input_source)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = num_frames / total_time if total_time > 0 else 0
        cap.release()

    # Clean up temp file
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    # Log metrics
    metrics = {
        "model_path": model_path,
        "input_source": input_source,
        "is_video": is_video,
        "output_dir": str(results[0].save_dir),
        "total_time_sec": round(total_time, 3),
        "fps": round(fps, 2) if fps else None,
        "total_detections": total_detections,
        "avg_confidence": round(avg_confidence, 3),
        "detection_summary": detection_summary
    }

    # Save metrics as JSON
    metrics_path = os.path.join(results[0].save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Print short summary
    print(f"[INFO] Inference complete in {metrics['total_time_sec']}s | "
          f"Detections: {total_detections} | Avg Conf: {metrics['avg_confidence']:.2f}")

    return metrics


if __name__ == "__main__":
    metrics = run_yolo_prediction(
        model_path='artifacts/models/runs/detect/train/weights/best.pt',
        input_source='data/processed/val/images/000000053624.jpg'
    )
    print(json.dumps(metrics, indent=4))




# 1. Load the YOLO model from the given model path.
# 2. Check if the input source is a file path or a file-like object.
# 3. If it’s a file-like object, write its contents to a temporary file and use that path.
# 4. Determine if the input is a video or an image by checking its file extension.
# 5. Start a timer to measure inference time.
# 6. Run the YOLO model’s `predict()` function using the input source, saving results to the output directory and applying the given confidence threshold.
# 7. Stop the timer once prediction completes and compute total inference time.
# 8. Initialize variables to track total number of detections, total confidence sum, and a detection summary list.
# 9. For each result returned by the model, get all bounding boxes and confidence scores.
# 10. If detections exist, add up all confidence values, count how many detections there are, compute the average confidence for that image, and store a summary entry with the image name, number of detections, and average confidence.
# 11. After processing all results, compute the overall average confidence across all detections.
# 12. If the input was a video, use OpenCV to open the video file, count the total number of frames, compute FPS as frames divided by total_time, and close the video.
# 13. If a temporary input file was created, delete it.
# 14. Collect all performance metrics into a dictionary, including model path, input path, whether it’s a video, output directory, total processing time, FPS (if video), total detections, average confidence, and per-image detection summary.
# 15. Save this metrics dictionary as a JSON file in the output directory.
# 16. Print a short summary of the run showing total time, total detections, and average confidence.
# 17. Return the metrics dictionary.

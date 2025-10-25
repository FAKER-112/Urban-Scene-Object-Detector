import os
import sys
import glob
import traceback
import gradio as gr
import streamlit as st
import tempfile
import shutil
from PIL import Image

# --- Add project root for imports ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
except Exception as e:
    print(f"Warning: Could not add project root to path. {e}")

# --- Import pipelines (with dummies for testing) ---
try:
    from src.pipeline.predict_pipeline import PredictionPipeline
    from src.pipeline.inference_pipeline import InferencePipeline
    from src.utils.logger import logger
    from src.utils.exception import CustomException
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}. Using dummy pipelines.")
    
    class CustomException(Exception): pass

    class PredictionPipeline:
        """Dummy PredictionPipeline for UI testing."""
        def __init__(self):
            self.working = True
            print("Warning: Dummy PredictionPipeline loaded.")
        def predict(self, filename):
            print(f"Dummy predict called for: {filename}")
            # Simulate creating an output file
            output_dir = tempfile.mkdtemp(prefix="single_out_")
            output_path = os.path.join(output_dir, os.path.basename(filename))
            shutil.copy2(filename, output_path)
            
            return {
                'total_detections': 10,
                'avg_confidence': 0.88,
                'total_time_sec': 1.23,
                'output_dir': output_dir,
                'is_video': os.path.splitext(filename)[1].lower() in SUPPORTED_VIDEO_EXT,
                'model_path': 'dummy/model.pt',
                'detection_summary': [{
                    'image': os.path.basename(filename),
                    'num_detections': 10,
                    'avg_confidence': 0.88
                }]
            }

    class InferencePipeline:
        """Dummy InferencePipeline for UI testing."""
        def __init__(self):
            self.input_dir = None
            self.output_dir = tempfile.mkdtemp(prefix="batch_out_")
            self.working = True
            print("Warning: Dummy InferencePipeline loaded.")
        
        def run(self):
            print(f"Dummy batch run called for input dir: {self.input_dir}")
            if not self.input_dir or not os.path.exists(self.input_dir):
                raise CustomException("Input directory not set or does not exist.")
            
            files = glob.glob(os.path.join(self.input_dir, "*.*"))
            for f in files:
                shutil.copy2(f, os.path.join(self.output_dir, os.path.basename(f)))
                
            return {
                'total_files': len(files),
                'successful_inferences': len(files),
                'failed_inferences': 0,
                'total_detections': len(files) * 5,
                'avg_confidence': 0.75,
                'total_runtime_sec': 5.67,
                'device': 'cpu (dummy)',
            }

    logger = type('Logger', (object,), {'info': print, 'error': print, 'warning': print, 'debug': print})()


# --- Constants & Configuration ---

SUPPORTED_IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
SUPPORTED_VIDEO_EXT = ['.mp4', '.avi', '.mov', '.mkv']

# --- CSS for Futuristic Look (Header Only) ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* Apply font to body - Streamlit handles the rest */
body {
    font-family: 'Inter', sans-serif;
}

.title-container {
    text-align: center;
    padding: 20px 0;
    border-bottom: 1px solid #334155;
}
.title-text {
    font-size: 2.5em;
    font-weight: 700;
    color: #E2E8F0;
    letter-spacing: -1px;
    background: -webkit-linear-gradient(45deg, #22D3EE, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.section-header {
    font-size: 1.5em;
    font-weight: 600;
    color: #94A3B8;
    border-bottom: 1px solid #334155;
    padding-bottom: 10px;
    margin-top: 20px;
}
/* Style for metric tables */
table {
    width: 100%;
    color: #E2E8F0;
}
th {
    color: #94A3B8;
}
code {
    color: #F472B6;
    background-color: #334155;
    padding: 2px 5px;
    border-radius: 4px;
}
</style>
"""

# --- Pipeline Initialization ---
@st.cache_resource
def load_pipelines():
    """Loads and caches the ML pipelines."""
    pred_pipe, infer_pipe = None, None
    try:
        pred_pipe = PredictionPipeline()
        logger.info("✅ Single prediction pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize single prediction pipeline: {e}")
        pred_pipe = None

    try:
        infer_pipe = InferencePipeline()
        logger.info("✅ Batch inference pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize batch inference pipeline: {e}")
        infer_pipe = None
    return pred_pipe, infer_pipe

prediction_pipeline, inference_pipeline = load_pipelines()

# --- Helper Functions ---

def save_uploaded_file(uploaded_file):
    """Saves an UploadedFile to a temporary path and returns the path."""
    try:
        # Use a named temporary file that persists
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return None

def find_output_file(output_dir, original_filename):
    """Find the output file in the prediction directory."""
    if not output_dir or not os.path.exists(output_dir):
        logger.warning(f"Output directory not found: {output_dir}")
        return None
    
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    logger.debug(f"Searching for output file based on base_name: {base_name}")

    # Try to find the exact match
    exact_match_pattern = os.path.join(output_dir, f"{base_name}.*")
    files = sorted(glob.glob(exact_match_pattern), key=os.path.getmtime, reverse=True)
    if files:
        logger.info(f"Found specific output file: {files[0]}")
        return files[0]

    # Fallback: Find the most recent file in the directory
    logger.warning(f"Could not find specific match for {base_name}. "
                   f"Falling back to most recent file in {output_dir}")
    all_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                 if os.path.isfile(os.path.join(output_dir, f))]
    if not all_files:
        logger.error(f"No files found in output directory: {output_dir}")
        return None
    
    latest_file = max(all_files, key=os.path.getctime)
    logger.info(f"Found latest file as fallback: {latest_file}")
    return latest_file

def format_single_metrics(metrics_data):
    """Formats single detection metrics into a Markdown string."""
    metrics_md = "## ✅ Analysis Complete\n\n"
    metrics_md += "| Metric | Value |\n| --- | --- |\n"
    for key, val in metrics_data.items():
        if key == "Detection Summary": continue
        metrics_md += f"| {key} | `{val}` |\n"
        
    if metrics_data.get("Detection Summary"):
        metrics_md += "\n### 📋 Detection Summary\n"
        for item in metrics_data["Detection Summary"]:
            metrics_md += (
                f"- **Image:** `{item.get('image', 'N/A')}`\n"
                f"  - **Detections:** {item.get('num_detections', 0)}\n"
                f"  - **Avg. Confidence:** {item.get('avg_confidence', 0):.3f}\n"
            )
    return metrics_md

def format_batch_metrics(metrics_data):
    """Formats batch detection metrics into a Markdown string."""
    metrics_md = "## ✅ Batch Processing Complete\n\n"
    metrics_md += "| Metric | Value |\n| --- | --- |\n"
    for key, val in metrics_data.items():
        metrics_md += f"| {key} | `{val}` |\n"
    
    metrics_md += f"\n### 📂 Output Location\n"
    metrics_md += f"Results saved to: `{metrics_data.get('Output Directory', 'N/A')}`\n"
    metrics_md += f"\nCheck the output directory for all annotated files and detailed metrics."
    return metrics_md

# --- SINGLE FILE PREDICTION LOGIC ---
def run_single_detection(input_filepath):
    """Single file detection logic."""
    if prediction_pipeline is None:
        return None, None, "Single prediction pipeline is not initialized. Please check server logs."
    
    try:
        logger.info(f"Processing single file: {input_filepath}")
        result = prediction_pipeline.predict(filename=input_filepath)
        logger.info(f"Prediction result: {result}")
        
        output_dir = result.get('output_dir')
        output_path = find_output_file(output_dir, input_filepath)

        if not output_path or not os.path.exists(output_path):
            return None, None, f"Detection completed but output file was not found in {output_dir}."

        metrics_dict = {
            "Total Detections": result.get('total_detections', 0),
            "Avg. Confidence": f"{result.get('avg_confidence', 0.0):.3f}",
            "Processing Time (s)": f"{result.get('total_time_sec', 0.0):.3f}",
            "Media Type": "Video" if result.get('is_video', False) else "Image",
            "Model Path": result.get('model_path', 'N/A'),
            "Output Directory": output_dir,
            "Detection Summary": result.get('detection_summary', [])
        }
        
        logger.info(f"Successfully processed single file. Output: {output_path}")
        return output_path, metrics_dict, None
    except Exception as e:
        logger.error(f"Unexpected error in single detection: {e}")
        logger.error(traceback.format_exc())
        return None, None, f"An unexpected server error occurred: {str(e)}"

# --- BATCH PREDICTION LOGIC ---
def run_batch_detection(input_filepaths, progress_bar):
    """Batch processing logic."""
    if inference_pipeline is None:
        return None, "Batch inference pipeline is not initialized. Please check server logs."
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="batch_inference_")
        logger.info(f"Created temporary directory: {temp_dir}")
        
        progress_bar.progress(0.1, text="Copying files...")
        for i, file_path in enumerate(input_filepaths):
            dest_path = os.path.join(temp_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
        
        logger.info(f"Processing {len(input_filepaths)} files in batch mode")
        
        original_input_dir = inference_pipeline.input_dir
        inference_pipeline.input_dir = temp_dir
        
        progress_bar.progress(0.3, text="Running batch inference...")
        batch_metrics = inference_pipeline.run()
        
        inference_pipeline.input_dir = original_input_dir
        
        progress_bar.progress(1.0, text="Complete!")
        
        metrics_dict = {
            "Total Files": batch_metrics.get('total_files', 0),
            "Successful": batch_metrics.get('successful_inferences', 0),
            "Failed": batch_metrics.get('failed_inferences', 0),
            "Total Detections": batch_metrics.get('total_detections', 0),
            "Avg. Confidence": f"{batch_metrics.get('avg_confidence', 0.0):.3f}",
            "Total Runtime (s)": f"{batch_metrics.get('total_runtime_sec', 0.0):.3f}",
            "Device": batch_metrics.get('device', 'N/A'),
            "Output Directory": inference_pipeline.output_dir,
        }
        
        return metrics_dict, None
    except Exception as e:
        logger.error(f"Unexpected error in batch: {e}")
        logger.error(traceback.format_exc())
        return None, f"An unexpected server error occurred: {str(e)}"
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")

# --- Main Streamlit App ---

def main():
    st.set_page_config(
        page_title="YOLOv8 Hyperion Interface",
        page_icon="🛰️",
        layout="wide"
    )
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state variables
    if 'single_result' not in st.session_state:
        st.session_state.single_result = {'path': None, 'metrics': None, 'error': None}
    if 'batch_result' not in st.session_state:
        st.session_state.batch_result = {'metrics': None, 'error': None}
    
    # --- Header ---
    st.markdown(
        """
        <div class='title-container'>
            <h1 class='title-text'>🚀 YOLOv8 Hyperion Interface 🛰️</h1>
            <p style='color: #94A3B8; margin-top: 5px;'>
            Single file or batch detection powered by YOLOv8
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["🖼️ Single File Detection", "📦 Batch Processing"])

    # --- SINGLE FILE TAB ---
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<h3 class='section-header'>1. Upload Media</h3>", unsafe_allow_html=True)
            
            single_input = st.file_uploader(
                "Upload Image or Video",
                type=SUPPORTED_IMAGE_EXT + SUPPORTED_VIDEO_EXT,
                accept_multiple_files=False,
                label_visibility="collapsed"
            )
            
            single_detect_btn = st.button("🚀 Run Detection", use_container_width=True, type="primary")
            
            st.markdown(
                """
                #### 📁 Supported Formats
                - **Images:** JPG, PNG, BMP, WEBP
                - **Videos:** MP4, AVI, MOV, MKV
                
                #### ℹ️ About
                Upload a single file for instant object detection.
                """
            )

        with col2:
            st.markdown("<h3 class='section-header'>2. Analysis & Results</h3>", unsafe_allow_html=True)
            
            if single_detect_btn and single_input:
                temp_path = save_uploaded_file(single_input)
                if temp_path:
                    with st.spinner("Processing..."):
                        path, metrics, error = run_single_detection(temp_path)
                        st.session_state.single_result = {'path': path, 'metrics': metrics, 'error': error}
                    
                    if os.path.exists(temp_path):
                        os.unlink(temp_path) # Clean up temp input file
            
            # Display results from session state
            result = st.session_state.single_result
            if result['error']:
                st.error(f"**Error:** {result['error']}")
            elif result['path']:
                is_image = os.path.splitext(result['path'])[1].lower() in SUPPORTED_IMAGE_EXT
                if is_image:
                    st.image(result['path'], caption="Detection Result", use_column_width=True)
                else:
                    st.video(result['path'])
                
                st.markdown("<h3 class='section-header'>3. Download & Metrics</h3>", unsafe_allow_html=True)
                
                try:
                    with open(result['path'], "rb") as f:
                        st.download_button(
                            "⬇️ Download Annotated File",
                            data=f,
                            file_name=os.path.basename(result['path']),
                            use_container_width=True
                        )
                except Exception as e:
                    st.warning(f"Could not read output file for download: {e}")
                
                st.markdown(format_single_metrics(result['metrics']), unsafe_allow_html=True)
                
            else:
                st.info("Upload a file and click 'Run Detection' to see results here.")
    
    # --- BATCH PROCESSING TAB ---
    with tab2:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<h3 class='section-header'>1. Upload Multiple Files</h3>", unsafe_allow_html=True)
            
            batch_input = st.file_uploader(
                "Upload Images/Videos (Multiple)",
                type=SUPPORTED_IMAGE_EXT + SUPPORTED_VIDEO_EXT,
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            batch_detect_btn = st.button("🚀 Run Batch Detection", use_container_width=True, type="primary")
            
            st.markdown(
                """
                #### 📁 Batch Processing
                - Upload multiple files at once
                - Parallel processing for faster results
                - All outputs saved to a directory
                
                #### ⚙️ Features
                - Multi-threaded execution
                - Progress tracking
                - Aggregated metrics
                """
            )

        with col2:
            st.markdown("<h3 class='section-header'>2. Batch Results</h3>", unsafe_allow_html=True)
            
            if batch_detect_btn and batch_input:
                temp_paths = [save_uploaded_file(f) for f in batch_input]
                temp_paths = [p for p in temp_paths if p is not None]
                
                if temp_paths:
                    progress_bar = st.progress(0, text="Starting batch...")
                    metrics, error = run_batch_detection(temp_paths, progress_bar)
                    st.session_state.batch_result = {'metrics': metrics, 'error': error}
                    
                    # Clean up temp input files
                    for p in temp_paths:
                        if os.path.exists(p):
                            os.unlink(p)
                else:
                    st.session_state.batch_result = {'metrics': None, 'error': "Failed to save uploaded files for processing."}

            # Display results from session state
            result = st.session_state.batch_result
            if result['error']:
                st.error(f"**Error:** {result['error']}")
            elif result['metrics']:
                st.markdown(format_batch_metrics(result['metrics']), unsafe_allow_html=True)
            else:
                st.info("Upload files and click 'Run Batch Detection' to see results here.")


if __name__ == "__main__":
    if prediction_pipeline is None and inference_pipeline is None:
        print("\n" + "="*50)
        print("❌ FATAL: Could not launch app because both pipelines failed to load.")
        print("Please check the logs above for errors.")
        print("="*50 + "\n")
        st.error("FATAL: Both pipelines failed to load. Check server logs.")
    else:
        status = []
        if prediction_pipeline: status.append("Single file detection ✅")
        if inference_pipeline: status.append("Batch processing ✅")
        print("\n" + "="*50)
        print("✅ App is ready to launch!")
        print(f"Available modes: {', '.join(status)}")
        print("="*50 + "\n")
        main()

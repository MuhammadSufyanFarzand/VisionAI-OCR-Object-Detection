import os
import cv2
import numpy as np
import urllib.request
import pytesseract
import streamlit as st
import time
import pandas as pd
from PIL import Image
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

# ==========================================
# 1. CORE PIPELINE & CONFIGURATION
# ==========================================

@dataclass
class AppConfig:
    execution_mode: str = "unified"
    prototxt_path: str = "MobileNetSSD_deploy.prototxt"
    model_path: str = "MobileNetSSD_deploy.caffemodel"
    object_confidence_threshold: float = 0.80
    text_confidence_threshold: int = 80
    
    # Active mirrors to download weights automatically if missing
    prototxt_url: str = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt"
    model_url: str = "https://github.com/nikmart/pi-object-detection/raw/master/MobileNetSSD_deploy.caffemodel"
    
    blob_scalefactor: float = 0.007843
    blob_size: Tuple[int, int] = (300, 300)
    blob_mean: Tuple[float, float, float] = (127.5, 127.5, 127.5)
    
    class_labels: List[str] = field(default_factory=lambda: [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ])


class UnifiedDetector:
    def __init__(self, config: AppConfig):
        self.config = config
        self._ensure_ssd_files()
        self.net = cv2.dnn.readNetFromCaffe(self.config.prototxt_path, self.config.model_path)

    def _ensure_ssd_files(self) -> None:
        """Downloads the required network graph and weights if they are missing."""
        for path, url in [(self.config.prototxt_path, self.config.prototxt_url), 
                          (self.config.model_path, self.config.model_url)]:
            if not os.path.exists(path):
                print(f"File '{path}' missing. Fetching from pre-trained model mirror...")
                try:
                    urllib.request.urlretrieve(url, path)
                    print(f"Successfully downloaded: {path}")
                except Exception as e:
                    print(f"Failed downloading {url}: {e}")
                    raise FileNotFoundError(f"Missing essential file: {path}")

    def detect_objects(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        h, w = image.shape[:2]
        output_image = image.copy()
        detected_objects = []
        
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=self.config.blob_scalefactor,
            size=self.config.blob_size,
            mean=self.config.blob_mean,
            swapRB=False,
            crop=False
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        total_predictions = detections.shape[2]
        
        for i in range(total_predictions):
            confidence = float(detections[0, 0, i, 2])
            class_idx = int(detections[0, 0, i, 1])
            
            # Strict 80% Filter Standard
            if confidence >= self.config.object_confidence_threshold:
                class_label = self.config.class_labels[class_idx] if class_idx < len(self.config.class_labels) else "Unknown"
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                
                detected_objects.append({
                    "label": class_label,
                    "confidence": confidence,
                    "box": (startX, startY, endX, endY)
                })
                
                color = (46, 204, 113)  # Bright Emerald Green
                cv2.rectangle(output_image, (startX, startY), (endX, endY), color, 3)
                label_text = f"{class_label.upper()} ({confidence:.1%})"
                cv2.putText(output_image, label_text, (startX, startY - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                
        return output_image, detected_objects

    def detect_and_recognize_text(self, image: np.ndarray, vis_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        # Pre-processing milestones
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Local Thresholding
        adaptive_thresholded = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
        
        custom_config = "--psm 6 --oem 3"
        detected_text_data = []
        
        try:
            data = pytesseract.image_to_data(adaptive_thresholded, config=custom_config, output_type=pytesseract.Output.DICT)
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                word_text = data['text'][i].strip()
                if not word_text or len(word_text) < 2:  # Skip single character noise
                    continue
                
                confidence_score = int(data['conf'][i])
                
                # Strict 80% filter for text
                if confidence_score >= self.config.text_confidence_threshold:
                    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    
                    detected_text_data.append({
                        "text": word_text,
                        "confidence": confidence_score / 100.0,
                        "box": (x, y, x + w, y + h)
                    })
                    
                    color = (52, 152, 219)  # Tech Blue
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(vis_image, word_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                        
        except Exception as e:
            print(f"OCR Error: {e}")
            
        return vis_image, adaptive_thresholded, detected_text_data


# ==========================================
# 2. INTERACTIVE STREAMLIT INTERFACE
# ==========================================

def run_analytical_dashboard(input_image: np.ndarray, mode: str, threshold: float) -> Tuple[np.ndarray, np.ndarray, str, List[List[Any]], str]:
    # Start latency calculation
    start_time = time.time()
    
    # input_image is loaded as RGB (Streamlit standard), convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    output_canvas = image_bgr.copy()
    
    config = AppConfig(
        execution_mode=mode.lower(),
        object_confidence_threshold=threshold,
        text_confidence_threshold=int(threshold * 100)
    )
    
    detector = UnifiedDetector(config)
    objects = []
    texts = []
    binarized_output = np.zeros_like(gray := cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))

    # Path 2: Object Detection execution
    if mode in ["Unified Framework (Both)", "Pre-trained Object Detection Only"]:
        output_canvas, objects = detector.detect_objects(output_canvas)
        
    # Path 1: OCR execution
    if mode in ["Unified Framework (Both)", "Pre-trained OCR Text Only"]:
        output_canvas, binarized_output, texts = detector.detect_and_recognize_text(image_bgr, output_canvas)

    # Convert canvases back to RGB for visualization in web elements
    annotated_rgb = cv2.cvtColor(output_canvas, cv2.COLOR_BGR2RGB)
    binarized_rgb = cv2.cvtColor(binarized_output, cv2.COLOR_GRAY2RGB) if len(binarized_output.shape) == 2 else binarized_output

    # End latency calculation
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    # Format statistical metrics panel
    metrics_panel_html = f"""
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px;'>
        <div style='background-color:#1e293b; padding:15px; border-radius:8px; text-align:center; color:white; border: 1px solid #334155;'>
            <p style='margin:0; font-size:11px; color:#94a3b8; text-transform:uppercase;'>Inference Latency</p>
            <p style='margin:5px 0 0 0; font-size:18px; font-weight:bold; color:#fbbf24;'>{latency_ms:.1f} ms</p>
        </div>
        <div style='background-color:#1e293b; padding:15px; border-radius:8px; text-align:center; color:white; border: 1px solid #334155;'>
            <p style='margin:0; font-size:11px; color:#94a3b8; text-transform:uppercase;'>Objects Detected</p>
            <p style='margin:5px 0 0 0; font-size:18px; font-weight:bold; color:#34d399;'>{len(objects)}</p>
        </div>
        <div style='background-color:#1e293b; padding:15px; border-radius:8px; text-align:center; color:white; border: 1px solid #334155;'>
            <p style='margin:0; font-size:11px; color:#94a3b8; text-transform:uppercase;'>Texts Extracted</p>
            <p style='margin:5px 0 0 0; font-size:18px; font-weight:bold; color:#60a5fa;'>{len(texts)}</p>
        </div>
    </div>
    """

    # Populate analytical tabular data logs
    tabular_logs = []
    for obj in objects:
        tabular_logs.append(["PHYSICAL ENTITY", obj["label"].upper(), f"{obj['confidence']:.1%}", str(obj["box"])])
    for txt in texts:
        tabular_logs.append(["WRITTEN TEXT", txt["text"], f"{txt['confidence']:.1%}", str(txt["box"])])

    extracted_sentences = " ".join([t["text"] for t in texts]) if texts else "No verified text blocks detected."

    return annotated_rgb, binarized_rgb, metrics_panel_html, tabular_logs, extracted_sentences


# ==========================================
# 3. STREAMLIT APP ENGINE
# ==========================================

# Set page configurations
st.set_page_config(
    page_title="Inference Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Global CSS for Dark Slate Theme
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .stButton>button {
        background: linear-gradient(135deg, #1abc9c, #16a085) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
    }
    .stButton>button:hover { background: #16a085 !important; }
    div[data-testid="stExpander"] { background-color: #1e293b !important; border: 1px solid #334155 !important; }
</style>
""", unsafe_allow_html=True)

# Standard Header Card
st.markdown("""
<div style='position: relative; text-align: center; padding: 25px 20px 20px 20px; background-color: #1e293b; border-radius: 12px; color: white; margin-bottom: 25px; border: 1px solid #334155;'>
    <!-- Left Corner Developer Badge -->
    <div style='position: absolute; top: 12px; left: 15px; font-size: 11px; font-family: monospace; font-weight: bold; color: #1abc9c; background-color: #0f172a; padding: 4px 10px; border-radius: 6px; border: 1px solid #334155;'>
        👤 SufyanDev
    </div>
    <h1 style='margin: 0; font-weight: 800; color: #1abc9c; letter-spacing: 1px;'>💻 EDGE-AI DASHBOARD PLATFORM</h1>
    <p style='margin: 5px 0 0 0; color: #cbd5e1; font-size: 14px;'>Unified pre-trained engine for structural Object Recognition and OCR character analysis with active validation filters.</p>
</div>
""", unsafe_allow_html=True)

# Divide screen into standard columns (scale=2 and scale=3 equivalent)
col1, col2 = st.columns([2, 3])

# Initialize session state for tabular logs persistence
if "st_tabular_logs" not in st.session_state:
    st.session_state["st_tabular_logs"] = []

with col1:
    st.markdown("<h3 style='color:#1abc9c; font-weight:bold; margin-bottom:15px;'>⚙️ Control Configurations</h3>", unsafe_allow_html=True)
    
    # Upload interface
    uploaded_file = st.file_uploader("Upload Diagnostic Source Image", type=["jpg", "jpeg", "png"])
    
    # Pipeline mode selector
    execution_mode = st.selectbox(
        "Processing Method",
        ["Unified Framework (Both)", "Pre-trained Object Detection Only", "Pre-trained OCR Text Only"]
    )
    
    # Strict threshold slider
    threshold = st.slider(
        "Confidence Threshold Benchmark (Strict Drop Filter)",
        min_value=0.10,
        max_value=1.00,
        value=0.80,
        step=0.05
    )
    
    run_btn = st.button("🚀 Run Inference Graph", use_container_width=True)

with col2:
    st.markdown("<h3 style='color:#1abc9c; font-weight:bold; margin-bottom:15px;'>📊 Real-time Analytical Outputs</h3>", unsafe_allow_html=True)
    
    if run_btn and uploaded_file is not None:
        # Load and convert image stream to NumPy
        image_pil = Image.open(uploaded_file)
        input_image = np.array(image_pil)
        
        # Execute processing loop
        annotated_rgb, binarized_rgb, metrics_html, logs, extracted_sentence = run_analytical_dashboard(
            input_image, execution_mode, threshold
        )
        
        # Display latency metrics HTML card
        st.markdown(metrics_html, unsafe_allow_html=True)
        
        # Create Output Tabs
        tab1, tab2 = st.tabs(["🖼️ Annotated Output View", "🔍 Pre-processed Binarized Image"])
        
        with tab1:
            st.image(annotated_rgb, caption="Annotated Visual Output (Green: Objects | Blue: Text)", use_container_width=True)
            
        with tab2:
            # Only show binarized details if OCR was active
            if execution_mode in ["Unified Framework (Both)", "Pre-trained OCR Text Only"]:
                st.image(binarized_rgb, caption="Binarized Adaptive Threshold Output", use_container_width=True)
            else:
                st.info("Binarization view is active only during OCR operations.")
                
        # Structured parsed text box
        st.text_area("📝 Extracted Sentence Block (OCR Engine)", value=extracted_sentence, height=110, disabled=False)
        
        # Save logs to session state
        st.session_state["st_tabular_logs"] = logs
        
    elif uploaded_file is None:
        st.info("Please upload an image from the control configurations panel on the left.")
    else:
        st.warning("Click 'Run Inference Graph' to launch the pre-trained deep learning networks.")

# Divider and Validation table section at the bottom
st.markdown("<hr style='border: 1px solid #334155; margin: 25px 0;'>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#1abc9c; font-weight:bold; margin-bottom:15px;'>📜 Structured Validation Logs</h3>", unsafe_allow_html=True)

if st.session_state["st_tabular_logs"]:
    # Render interactive clean dataframe log table
    logs_df = pd.DataFrame(
        st.session_state["st_tabular_logs"],
        columns=["Detection Domain", "Label/Extracted Text", "Confidence Value", "Bounding Box Coordinates"]
    )
    st.dataframe(logs_df, use_container_width=True, hide_index=True)
else:
    st.text("No active validation logs. Please run a diagnostic scan above.")
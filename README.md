# Dcodelabes Project 4
---

title: Intelligent Vision Dashboard
emoji: 💻
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Intelligent Vision Dashboard

An AI-powered Computer Vision application that combines **Optical Character Recognition (OCR)** and **Object Detection** into a unified intelligent vision system. The application leverages **Tesseract OCR** for text extraction and **MobileNet-SSD** for real-time object detection, providing fast and accurate analysis through an interactive web interface.

> **Developed during my AI Internship at Decode Labs**

---

# Project Overview

The Intelligent Vision Dashboard demonstrates an end-to-end computer vision pipeline capable of extracting textual information and identifying objects from images using industry-standard deep learning models.

The system integrates image preprocessing techniques with pretrained neural networks to improve recognition accuracy while maintaining lightweight and efficient inference performance.

This project was developed as part of my **Artificial Intelligence Internship at Decode Labs**, where the focus was on implementing practical Computer Vision solutions using modern AI technologies.

---

# Key Features

- Intelligent OCR using Tesseract OCR Engine
- Real-Time Object Detection using MobileNet-SSD
- Interactive Web Interface
- Image Upload Support
- Confidence Score Filtering (80%)
- Automatic Image Preprocessing
- Bounding Box Visualization
- Text Extraction from Images
- Lightweight Deep Learning Inference
- Fast Processing Pipeline

---

# Technologies Used

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Computer Vision | OpenCV |
| OCR Engine | Tesseract OCR |
| OCR Wrapper | pytesseract |
| Deep Learning | MobileNet-SSD |
| Numerical Computing | NumPy |
| Image Processing | Pillow |
| Web Framework | Streamlit |
| Neural Network Backend | OpenCV DNN |

---

# Project Architecture

```
VisionAI-OCR-Object-Detection
│
├── app.py
├── requirements.txt
├── README.md
│
├── images/
├── outputs/
├── samples/
│
└── models/
    ├── MobileNetSSD_deploy.prototxt
    └── MobileNetSSD_deploy.caffemodel
```

---

# System Workflow

```
                   Input Image
                        │
                        ▼
             Image Preprocessing
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
   OCR Pipeline                 Object Detection
        │                               │
        ▼                               ▼
 Extract Text                  Detect Objects
        │                               │
        └───────────────┬───────────────┘
                        ▼
             Confidence Filtering
                        │
                        ▼
               Display Final Results
```

---

# Image Preprocessing Pipeline

Before analysis, every uploaded image undergoes several preprocessing stages to improve recognition accuracy.

The preprocessing pipeline includes:

- Grayscale Conversion
- Gaussian Blur
- Adaptive Thresholding
- Noise Reduction

These techniques significantly enhance OCR performance by improving text visibility and reducing image artifacts.

---

# OCR Module

The OCR component utilizes **Google Tesseract OCR** through the Python wrapper **pytesseract**.

### Supported Capabilities

- Printed Text Recognition
- Document Digitization
- Invoice Processing
- Image-to-Text Conversion
- Text Extraction from Photographs

---

# Object Detection Module

Object detection is implemented using the pretrained **MobileNet-SSD** architecture through OpenCV's Deep Neural Network (DNN) module.

### Detection Features

- Real-Time Object Detection
- Bounding Box Generation
- Object Class Prediction
- Confidence Score Calculation
- Lightweight Deep Learning Inference

---

# Confidence Threshold

To improve prediction reliability, only detections with a confidence score greater than **80%** are displayed.

```python
if confidence >= 0.80:
    draw_bounding_box()
else:
    ignore_prediction()
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/VisionAI-OCR-Object-Detection.git
```

## Navigate to Project Directory

```bash
cd VisionAI-OCR-Object-Detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

---

# Example Use Cases

- Document Digitization
- Invoice Processing
- Smart Document Reading
- OCR-Based Automation
- Object Recognition
- Educational Computer Vision Projects
- AI Learning Applications

---

# Future Enhancements

- EasyOCR Integration
- YOLOv11 Object Detection
- Multi-language OCR
- PDF OCR Support
- Webcam Live Detection
- Batch Image Processing
- Export Results to CSV
- Export Reports to PDF
- Cloud Deployment
- REST API Integration

---

# Learning Outcomes

This project strengthened my practical understanding of:

- Computer Vision
- Deep Learning Inference
- Image Processing
- OCR Systems
- OpenCV
- MobileNet-SSD
- Tesseract OCR
- Python Development
- AI Application Deployment
- Streamlit Development

---

# Internship Information

This project was successfully completed during my **Artificial Intelligence Internship at Decode Labs**.

### Organization

**Decode Labs**

### Internship Domain

Artificial Intelligence (AI)

### Project Title

**Intelligent Vision Dashboard**

### Project Focus

- Optical Character Recognition (OCR)
- Object Detection
- Image Processing
- Computer Vision
- Deep Learning Applications

This internship provided hands-on experience in developing production-style AI applications using modern Computer Vision techniques and industry-standard Python libraries.

---

# Author

## Muhammad Sufyan

**AI Developer | Computer Vision Engineer | LLM Engineer | Agentic AI Developer | RAG Developer | FastAPI | LangChain | LangGraph | Python**

### Connect with Me

**GitHub**

https://github.com/MuhammadSufyanFarzand

**LinkedIn**

https://www.linkedin.com/in/muhammad-sufyan-farzand-096a3b377/

---

# License

This project was developed for educational, research, and portfolio purposes as part of my AI Internship at **Decode Labs**.

The project may be used for learning and academic reference with proper attribution.

---

## Acknowledgements

Special thanks to **Decode Labs** for providing the internship opportunity, mentorship, and practical learning environment that enabled the successful development of this Computer Vision project.

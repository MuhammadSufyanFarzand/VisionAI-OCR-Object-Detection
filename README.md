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
Developed by sufyan dev. A unified pre-trained engine for structural Object Recognition (MobileNet-SSD) and OCR character analysis (Tesseract LSTM) with active 80% confidence threshold filters.
# VisionAI OCR & Object Detection

An AI-powered Computer Vision application that performs **Optical Character Recognition (OCR)** and **Object Detection** using industry-standard computer vision libraries. The application combines intelligent image preprocessing with pre-trained deep learning models to extract text and detect objects from images in real time.

---

## Overview

This project was developed as **Project 4** during my AI Internship at **DecodeLabs**.

The system demonstrates the complete computer vision pipeline:

- Image Preprocessing
- OCR using Tesseract
- Object Detection using MobileNet-SSD
- Confidence Score Filtering
- Bounding Box Visualization
- Interactive Web Interface with Streamlit

---

## Features

- Image Upload
- OCR Text Extraction
- MobileNet-SSD Object Detection
- Confidence Threshold Filtering (80%)
- Image Preprocessing
  - Grayscale Conversion
  - Gaussian Blur
  - Adaptive Thresholding
- Bounding Box Visualization
- Streamlit Web Interface
- Fast and Lightweight Inference

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| OpenCV | Image Processing |
| Tesseract OCR | Text Recognition |
| pytesseract | Python OCR Wrapper |
| MobileNet-SSD | Object Detection |
| NumPy | Numerical Computing |
| Pillow | Image Handling |
| Streamlit | Web Application |

---

## Project Structure

```
VisionAI-OCR-Object-Detection
│
├── app.py
├── requirements.txt
├── README.md
├── images/
├── models/
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
├── samples/
└── outputs/
```

---

## Workflow

```
Input Image
      │
      ▼
Image Preprocessing
      │
      ▼
───────────────
Choose Pipeline
───────────────
      │
 ┌────┴────┐
 │         │
 ▼         ▼
 OCR    Object Detection
 │         │
 ▼         ▼
Extract Text   Detect Objects
 │         │
 └────┬────┘
      ▼
Confidence Filtering
      ▼
Display Results
```

---

## Image Preprocessing

Before recognition, the image passes through several preprocessing stages:

- Grayscale Conversion
- Gaussian Blur
- Adaptive Thresholding
- Noise Reduction

These steps significantly improve OCR accuracy.

---

## OCR Pipeline

The OCR module uses Google's **Tesseract OCR Engine** through the `pytesseract` Python wrapper.

Supported Features:

- Printed Text Recognition
- Document OCR
- Invoice Reading
- Image Text Extraction

---

## Object Detection

Object detection is powered by **MobileNet-SSD** using OpenCV's DNN module.

Features include:

- Real-time Detection
- Bounding Boxes
- Confidence Scores
- Class Labels

---

## Confidence Threshold

Only predictions with confidence greater than **80%** are displayed.

```python
if confidence >= 0.80:
    draw_box()
else:
    ignore_prediction()
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/VisionAI-OCR-Object-Detection.git
```

Move into the project

```bash
cd VisionAI-OCR-Object-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Future Improvements

- EasyOCR Integration
- YOLOv11 Support
- PDF OCR
- Multi-language OCR
- Webcam Detection
- Batch Image Processing
- Export Results as CSV/PDF

---

## Learning Outcomes

This project strengthened my understanding of:

- Computer Vision
- OCR Systems
- OpenCV
- Image Processing
- Deep Learning Inference
- MobileNet-SSD
- Streamlit Deployment
- AI Application Development

---

## Author

**Muhammad Sufyan**

AI Developer | Computer Vision | LLMs | Agentic AI | RAG | FastAPI | LangChain

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

---

## License

This project is intended for educational and research purposes.

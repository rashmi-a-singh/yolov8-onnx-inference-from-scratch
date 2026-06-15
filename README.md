# YOLOv8 ONNX Inference Pipeline — From Scratch

Manual implementation of the full YOLOv8 inference pipeline using ONNX Runtime.
No Ultralytics wrapper used for inference — every step written explicitly.

## What this covers

| Section | Content |
|---|---|
| 1 — Image Manipulation | BGR/RGB, grayscale, letterbox, normalisation, bounding boxes |
| 2 — ONNX Export | YOLOv8n / s / m export, model size comparison, graph inspection |
| 3 — Manual Inference | Letterbox → normalise → CHW → batch → ONNX → decode → NMS → draw |
| 4 — Benchmarking | FPS, latency, model size comparison across variants |

## Why manual?

Edge deployment (Jetson, IP cameras, smart devices) cannot run PyTorch.
The model ships as an `.onnx` file and runs via ONNX Runtime — a lightweight
C++ engine with no framework dependency. Understanding every step of
preprocessing and postprocessing is essential for debugging production pipelines.

## Key concepts implemented

- **Letterboxing** — aspect-ratio-preserving resize with grey padding (YOLO standard)
- **HWC → CHW transpose** — OpenCV layout to PyTorch/ONNX layout
- **Raw output decoding** — `(1, 84, 8400)` → centre-format boxes + class scores
- **Manual NMS** — per-class Non-Maximum Suppression via `cv2.dnn.NMSBoxes`
- **Inverse letterbox transform** — mapping predicted coords back to original image space

## Stack

Python · OpenCV · NumPy · ONNX Runtime · Ultralytics (export only) · Matplotlib

## Background

Built alongside production CV work at LSE — a 3-stage pedestrian detection pipeline
(YOLOv11 → EfficientNet-B0 rider gate → EfficientNet-B3 + YOLOv11M ensemble)
processing 1M+ Google Street View images on Kaggle GPUs.

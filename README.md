# Fabric Defect Detection Using YOLOv8

A professional AI-based system to detect defects in fabric images using the YOLOv8 deep learning model. This project includes a complete training pipeline, prediction script, dataset configuration, and optimizations for low-VRAM GPUs like the RTX 3050.

# Features

YOLOv8-L model for high-accuracy defect detection

Supports custom annotated datasets

VRAM-optimized training (safe for 4GB GPUs)

Advanced augmentations for improved generalization

Modular training and prediction scripts

Ready for integration into Django/React Native applications

# Project Structure

Fabric-Defect-Detection/
│
├── train.py
├── predict.py
├── README.md
│
└── data/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

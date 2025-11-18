# Fabric Defect Detection Using YOLOv8

A professional AI-based system to detect defects in fabric images using the YOLOv8 deep learning model. This project includes a complete training pipeline, prediction script, dataset configuration, and optimizations for low-VRAM GPUs like the RTX 3050.

# Features

- YOLOv8-L model for high-accuracy defect detection
- Supports custom annotated datasets
- VRAM-optimized training (safe for 4GB GPUs)
- Advanced augmentations for improved generalization
- Modular training and prediction scripts
- Ready for integration into Django/React Native applications

# Project Structure
```
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
        └── val
```
# Project Workflow
```
              ┌───────────────────────┐
              │   Dataset Collection   │
              └────────────┬──────────┘
                           ▼
                 ┌──────────────────┐
                 │  Manual Annotation│ (LabelImg/Roboflow)
                 └──────────┬───────┘
                            ▼
               ┌──────────────────────┐
               │   YOLO format data   │
               └───────────┬─────────┘
                           ▼
          ┌────────────────────────────────┐
          │       Training (YOLOv8)        │
          │  - Augmentations               │
          │  - AdamW Optimizer             │
          │  - 640×640 Resolution          │
          └───────────────┬────────────────┘
                          ▼
             ┌──────────────────────────┐
             │   Model Evaluation        │
             │ Precision / Recall / mAP  │
             └──────────────┬───────────┘
                            ▼
                ┌────────────────────┐
                │ Real-time Inference │
                └────────────────────┘

```
# Installation
```
pip install ultralytics torch scipy
```
# Dataset YAML (data.yaml)
```
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 8
names: ['Holes', 'Tails out', 'broken pick', 'omitted pick', 'others', 'slubs', 'strains', 'streaks or shading']

roboflow:
  workspace: project-zkzkn
  project: annotation-kgjqq
  version: 4
  license: CC BY 4.0
  url: https://universe.roboflow.com/project-zkzkn/annotation-kgjqq/dataset/4
```
# Training Script (train.py)
```python
from ultralytics import YOLO
import torch, gc

# -------------------------------------------------------------
# FUNCTION: clear_vram
# -------------------------------------------------------------
# • Clears Python garbage collection
# • Empties CUDA GPU VRAM
# • Prevents "CUDA OUT OF MEMORY" errors on low-VRAM GPUs (e.g., RTX 3050)
# -------------------------------------------------------------
def clear_vram():
    """Clear GPU VRAM & Python garbage memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[INFO] VRAM Cleared Successfully!")


# -------------------------------------------------------------
# FUNCTION: train_yolo
# -------------------------------------------------------------
# • Loads pretrained YOLOv8-L model (best for detection accuracy)
# • Trains using custom dataset defined in data.yaml
# • Includes optimized hyperparameters for fabric defect detection
# • Includes strong augmentations for better generalization
# • Optimized for 4GB VRAM GPUs using safe batch size & workers
# -------------------------------------------------------------
def train_yolo():
    # Step 1: Clear VRAM before starting training
    clear_vram()

    # Step 2: Load strong-base YOLOv8-L model
    model = YOLO("yolov8l.pt")

    # Step 3: Begin training with custom parameters
    model.train(
        data="data/data.yaml",     # Path to dataset configuration
        epochs=33,                 # Total training epochs
        imgsz=640,                 # Training image size
        batch=8,                   # Safe batch size for 4GB GPU
        workers=2,                 # Number of dataloader workers
        optimizer="AdamW",         # Stable optimizer for defect detection
        lr0=0.0005,                # Initial learning rate
        weight_decay=0.0003,       # Prevents overfitting
        patience=20,               # Early stopping patience

        # ----- Data Augmentation Settings -----
        mosaic=0.8,                # Mosaic augmentation
        mixup=0.1,                 # Mixup augmentation
        hsv_h=0.02,                # Hue variation
        hsv_s=0.6,                 # Saturation variation
        hsv_v=0.3,                 # Brightness variation
        scale=0.8,                 # Random image scaling
        fliplr=0.5,                # Horizontal flip probability

        cache=True,                # Cache images for faster training
        device=0                   # Use GPU (0)
    )


# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------
# When ran as a script:
# • VRAM clears → YOLO loads → Training starts
# -------------------------------------------------------------
if __name__ == "__main__":
    train_yolo()
```
# Prediction Script (predict.py)
```
from ultralytics import YOLO

def predict(image_path):
    model = YOLO("runs/detect/train/weights/best.pt")
    results = model(image_path)
    results.show()
    results.save("predictions/")

if __name__ == "__main__":
    predict("test.jpg")
```

# Output Files (Auto-generated by YOLO)
```
runs/detect/train/
│── weights/
│     ├── best.pt
│     └── last.pt
│── results.png
│── confusion_matrix.png
│── PR_curve.png
│── F1_curve.png
│── labels_correlogram.png
```
# Performance Metrics

Typical scores for a good dataset:
```
Metric	Expected Range
Precision	0.85 – 0.92
Recall	0.80 – 0.90
mAP50	0.82 – 0.94
```
# Hardware Optimization (RTX 3050)
~~~Parameter	Value
Batch Size	8
Image Size	640
Workers	2
Optimizer	AdamW
Device	GPU (0)
~~~
```
# Future Improvements

Add YOLOv10 for faster inference

Add Grad-CAM heatmap visualization

Build a Django + React Native full app

Deploy using FastAPI / Streamlit

# References

Ultralytics YOLOv8 Documentation

Redmon et al. “You Only Look Once”

Research papers on textile defect detection

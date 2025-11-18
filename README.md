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


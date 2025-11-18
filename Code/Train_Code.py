from ultralytics import YOLO
import torch, gc, os

def clear_vram():
    """
    Clears GPU VRAM and Python garbage.
    Helps prevent CUDA out-of-memory errors during training.
    """
    gc.collect()  # Clears Python garbage

    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # Clears unused VRAM
        torch.cuda.ipc_collect()   # Cleans inter-process cache
        print("[INFO] 🧹 VRAM cleared successfully")


def train_new_model():
    """
    Trains a new YOLOv8 model from scratch/fine-tune.
    Includes good hyperparameters for fabric defect detection.
    """
    clear_vram()

    # ----------------------
    # 1. Dataset & Model Path
    # ----------------------
    data_yaml = r"D:\manual annoated\data.yaml"   # Path to your dataset YAML file
    pretrained_model = "yolov8l.pt"               # Using YOLOv8-Large for higher accuracy

    # ----------------------
    # 2. Check Dataset File
    # ----------------------
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"❌ Dataset YAML not found at: {data_yaml}")

    # ----------------------
    # 3. Load Pretrained YOLO Model
    # ----------------------
    model = YOLO(pretrained_model)

    # ----------------------
    # 4. Train Your Model
    # ----------------------
    results = model.train(
        data=data_yaml,          # Dataset description
        epochs=33,               # Training epochs
        imgsz=640,               # Training image size
        batch=8,                 # Fits in 4GB VRAM (RTX 3050)
        workers=2,               # Dataloader workers
        optimizer="AdamW",       # Better optimizer
        lr0=0.0005,              # Initial learning rate
        patience=20,             # Early stopping patience
        weight_decay=0.0003,     # Regularization to reduce overfitting

        # ---- Augmentations ----
        mosaic=0.8,              # Strong mosaic for defect variety
        mixup=0.1,               # Slight mixup
        hsv_h=0.02,              # Color augments
        hsv_s=0.6,
        hsv_v=0.3,
        degrees=3,               # Small rotation
        translate=0.05,          # Small translation
        scale=0.8,               # Scaling
        fliplr=0.5,              # 50% horizontal flip

        cache=True,              # Caches dataset to RAM for speed
        device=0,                # GPU index
        resume=False             # Always start fresh
    )

    # ----------------------
    # 5. Training Complete
    # ----------------------
    print("\n✅ Training completed successfully!")
    print(f"📁 Check results here: {os.path.join(os.getcwd(), 'runs/detect/train')}")


# ----------------------
# Main Entry Point
# ----------------------
if __name__ == "__main__":
    train_new_model()

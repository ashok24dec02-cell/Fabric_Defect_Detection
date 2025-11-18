from ultralytics import YOLO
import cv2
import os

def run_prediction(image_path, output_path="output.jpg"):
    """
    Runs YOLOv8 inference on a single image.
    - image_path: Path to the input fabric image
    - output_path: Path to save annotated output
    """

    # ----------------------
    # 1. Load Your Trained Model
    # ----------------------
    model = YOLO("best.pt")   # Replace with correct path if needed

    # ----------------------
    # 2. Run Detection
    # ----------------------
    results = model(image_path)

    # ----------------------
    # 3. Save Annotated Image
    # ----------------------
    # YOLO automatically generates annotated results in results[0].plot()
    annotated_img = results[0].plot()

    cv2.imwrite(output_path, annotated_img)
    print(f"✅ Prediction complete! Annotated image saved at: {output_path}")

    # Optional: Print detected defects
    for box in results[0].boxes:
        cls = int(box.cls[0])        # Class ID
        conf = float(box.conf[0])    # Confidence
        print(f"Detected {model.names[cls]} with confidence: {conf:.2f}")


# ----------------------
# Example Usage
# ----------------------
if __name__ == "__main__":
    run_prediction("test.jpg", "annotated_output.jpg")

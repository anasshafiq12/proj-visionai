import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO  # Import YOLO from ultralytics

# Load YOLOv8 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov8s.pt")  # Load pretrained YOLOv8 small model
model.to(device)
model.eval()

# Enable half-precision inference if using CUDA
if device.type == "cuda":
    model.model.half()  # Access the underlying PyTorch model for half-precision

def detect_objects(frame):
    """
    Optimized YOLOv8 object detection.

    Args:
        frame (numpy.ndarray): The input image.

    Returns:
        list: A list of detected objects with labels and coordinates.
    """
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert image to PIL format
        img_pil = Image.fromarray(rgb_frame)

        # Run inference
        with torch.no_grad():
            results = model(img_pil)  # Run detection

        detected_objects = []

        # Extract detection results
        for det in results[0].boxes:  # Access boxes from first result
            detected_objects.append({
                "label": model.names[int(det.cls)],  # Class name from index
                "confidence": float(det.conf),  # Confidence score
                "bbox": (int(det.xyxy[0][0]), int(det.xyxy[0][1]), 
                         int(det.xyxy[0][2]), int(det.xyxy[0][3]))  # Bounding box
            })

        return detected_objects

    except Exception as e:
        print(f"Error in object detection: {str(e)}")
        return []

# Test function
if __name__ == "__main__":
    frame = cv2.imread("test_image.jpg")  # Load a test image
    results = detect_objects(frame)
    print(results)  # Print detected objects
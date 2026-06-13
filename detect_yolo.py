from ultralytics import YOLO
import cv2
import os

INPUT_IMAGE = "images/test1.jpg"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

results = model(INPUT_IMAGE)

annotated_image = results[0].plot()

output_path = os.path.join(OUTPUT_DIR, "06_yolo_vehicle_detection.jpg")
cv2.imwrite(output_path, annotated_image)

print(f"YOLO detekcija završena. Rezultat spremljen u: {output_path}")

from ultralytics import YOLO
import cv2
import os

INPUT_IMAGE = "images/test1.jpg"
MODEL_PATH = "models/license_plate_detector.pt"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Nedostaje model: models/license_plate_detector.pt"
    )

model = YOLO(MODEL_PATH)

results = model(INPUT_IMAGE)

annotated_image = results[0].plot()

output_path = os.path.join(OUTPUT_DIR, "07_yolo_license_plate_detection.jpg")
cv2.imwrite(output_path, annotated_image)

print(f"Detekcija registarske pločice završena: {output_path}")

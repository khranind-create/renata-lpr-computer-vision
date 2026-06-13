from ultralytics import YOLO
import cv2
import easyocr
import os

INPUT_IMAGE = "images/test1.jpg"
MODEL_PATH = "models/license_plate_detector.pt"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(["en"])

image = cv2.imread(INPUT_IMAGE)
results = model(INPUT_IMAGE)

for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    plate_crop = image[y1:y2, x1:x2]
    crop_path = f"{OUTPUT_DIR}/08_plate_crop_{i}.jpg"
    cv2.imwrite(crop_path, plate_crop)

    ocr_results = reader.readtext(plate_crop)

    plate_text = ""
    for result in ocr_results:
        plate_text += result[1] + " "

    plate_text = plate_text.strip()

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        plate_text if plate_text else "N/A",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    print(f"Pločica {i + 1}: {plate_text}")
    print(f"Crop spremljen: {crop_path}")

output_path = f"{OUTPUT_DIR}/09_ocr_result.jpg"
cv2.imwrite(output_path, image)

print(f"OCR rezultat spremljen u: {output_path}")
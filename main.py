import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_IMAGE = "images/test1.jpg"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image = cv2.imread(INPUT_IMAGE)

if image is None:
    raise FileNotFoundError(f"Slika nije pronađena: {INPUT_IMAGE}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Bilateralni filter
bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
bilateral_rgb = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)

# 2. Morfološke operacije
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)

morph_gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

# 3. Normalizacija svjetline
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
y_channel, cr, cb = cv2.split(ycrcb)

equalized_y = cv2.equalizeHist(y_channel)
normalized = cv2.merge((equalized_y, cr, cb))
normalized = cv2.cvtColor(normalized, cv2.COLOR_YCrCb2BGR)
normalized_rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)

# Spremanje pojedinačnih slika
cv2.imwrite(f"{OUTPUT_DIR}/01_original.jpg", image)
cv2.imwrite(f"{OUTPUT_DIR}/02_bilateral_filter.jpg", bilateral)
cv2.imwrite(f"{OUTPUT_DIR}/03_morphological_operations.jpg", morph_gradient)
cv2.imwrite(f"{OUTPUT_DIR}/04_brightness_normalization.jpg", normalized)

# Usporedni prikaz
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Originalna slika")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(bilateral_rgb)
plt.title("Bilateralni filter")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(morph_gradient, cmap="gray")
plt.title("Morfološke operacije")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(normalized_rgb)
plt.title("Normalizacija svjetline")
plt.axis("off")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_comparison.png", dpi=300)
plt.show()

print("Obrada završena. Rezultati su spremljeni u folder output.")
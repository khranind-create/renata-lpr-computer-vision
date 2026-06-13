import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

methods = [
    "Bilateralni filter",
    "Morfološke operacije",
    "Normalizacija svjetline"
]

precision = [91.8, 89.9, 93.4]
recall = [89.7, 88.4, 91.6]
f1_score = [90.7, 89.1, 92.5]
iou = [84.2, 82.7, 86.8]

# 1. F1-score grafikon
plt.figure(figsize=(9, 5))
plt.bar(methods, f1_score)
plt.title("Usporedba F1-score vrijednosti po metodama predobrade")
plt.ylabel("F1-score (%)")
plt.ylim(80, 100)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_f1_score_comparison.png", dpi=300)
plt.close()

# 2. Precision, Recall i F1 usporedba
x = range(len(methods))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar([i - width for i in x], precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar([i + width for i in x], f1_score, width, label="F1-score")

plt.title("Usporedba Precision, Recall i F1-score metrika")
plt.ylabel("Vrijednost (%)")
plt.ylim(80, 100)
plt.xticks(x, methods, rotation=15)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_detection_metrics_comparison.png", dpi=300)
plt.close()

# 3. IoU grafikon
plt.figure(figsize=(9, 5))
plt.bar(methods, iou)
plt.title("Usporedba IoU vrijednosti po metodama predobrade")
plt.ylabel("IoU (%)")
plt.ylim(75, 95)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_iou_comparison.png", dpi=300)
plt.close()

print("Grafikoni su generirani i spremljeni u output folder.")
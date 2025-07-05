import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# === CONFIG ===
csv_path       = r"runs\detect\train\results.csv"
images_dir     = r"runs\detect\train"
MODEL_PATH     = "runs/detect/train/weights/best.pt"
VAL_IMG_DIR    = Path("C:/Users/rageb/Desktop/new yolo model/datasets/valid_filtered/images")
VAL_LABEL_DIR  = Path("C:/Users/rageb/Desktop/new yolo model/datasets/valid_filtered/labels")
CLASS_NAMES    = ['car', 'emv', 'htv']
IOU_THRESH     = 0.5
CONF_THRESH    = 0.25
SAVE_DIR       = Path("metrics")
SAVE_DIR.mkdir(exist_ok=True)

# === Load YOLO model ===
model = YOLO(MODEL_PATH)

# === Helpers ===
def xywh2xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    return [x1, y1, x2, y2]

def compute_iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter_w, inter_h = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = inter_w * inter_h
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0

# === Confusion Matrix Generation ===
num_classes = len(CLASS_NAMES)
bg_idx = num_classes - 1
y_true, y_pred = [], []

for img_path in tqdm(list(VAL_IMG_DIR.rglob("*.jpg")), desc="Evaluating"):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    results = model(img, conf=CONF_THRESH)[0]
    pred_boxes, pred_cls, pred_scores = [], [], []
    if results.boxes is not None:
        for box in results.boxes:
            pred_boxes.append(box.xyxy.cpu().numpy()[0].tolist())
            pred_cls.append(int(box.cls.cpu().numpy()[0]))
            pred_scores.append(float(box.conf.cpu().numpy()[0]))

    gt_file = VAL_LABEL_DIR / f"{img_path.stem}.txt"
    if not gt_file.exists():
        continue

    gt_boxes, gt_cls = [], []
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            xc, yc, ww, hh = map(float, parts[1:5])
            gt_cls.append(cid)
            gt_boxes.append(xywh2xyxy(xc, yc, ww, hh, w, h))

    matched_gt, matched_pred = set(), set()
    preds = sorted(
        zip(pred_boxes, pred_cls, pred_scores, range(len(pred_boxes))),
        key=lambda x: x[2], reverse=True
    )
    for pb, pc, ps, pidx in preds:
        best_i, best_iou = -1, 0.0
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_i = iou, gi
        if best_iou >= IOU_THRESH:
            y_true.append(gt_cls[best_i])
            y_pred.append(pc)
            matched_gt.add(best_i)
            matched_pred.add(pidx)

    for pidx, pc in enumerate(pred_cls):
        if pidx not in matched_pred:
            y_true.append(bg_idx)
            y_pred.append(pc)

    for gi, gc in enumerate(gt_cls):
        if gi not in matched_gt:
            y_true.append(gc)
            y_pred.append(bg_idx)

cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
disp = ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix")
plt.savefig(SAVE_DIR / "confusion_matrix_normalized.png")
plt.close()

# === Plot Training/Validation Stats ===
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Losses
plt.figure(figsize=(12, 8))
plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")
plt.plot(df["epoch"], df["train/cls_loss"], label="Class Loss")
plt.plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss")
plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
plt.plot(df["epoch"], df["val/cls_loss"], label="Val Class Loss")
plt.plot(df["epoch"], df["val/dfl_loss"], label="Val DFL Loss")
plt.title("Training and Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(SAVE_DIR / "losses_plot.png")
plt.close()

# Metrics
plt.figure(figsize=(12, 8))
plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5")
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
plt.title("Eval Metrics")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig(SAVE_DIR / "metrics_plot.png")
plt.close()

# Learning Rates
plt.figure(figsize=(12, 8))
plt.plot(df["epoch"], df["lr/pg0"], label="lr/pg0")
plt.plot(df["epoch"], df["lr/pg1"], label="lr/pg1")
plt.plot(df["epoch"], df["lr/pg2"], label="lr/pg2")
plt.title("Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.legend()
plt.grid(True)
plt.savefig(SAVE_DIR / "learning_rates_plot.png")
plt.close()

# === Display Pre-generated Curves (if available) ===
image_files = [
    "F1_curve.png",
    "P_curve_.png",
    "P_curve.png",
    "PR_curve.png",
    "R_curve.png",
]

for filename in image_files:
    src_path = os.path.join(images_dir, filename)
    dst_path = SAVE_DIR / filename
    if os.path.exists(src_path):
        img = mpimg.imread(src_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(filename.replace(".png", "").replace("_", " ").title())
        plt.savefig(dst_path)
        plt.close()
    else:
        print(f"[Warning] {filename} not found in {images_dir}")

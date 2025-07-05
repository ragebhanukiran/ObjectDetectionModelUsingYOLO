import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultralytics import YOLO
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, auc,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
from torch.utils.tensorboard import SummaryWriter

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH     = "runs/detect/train/weights/best.pt"
VAL_IMG_DIR    = Path("C:/Users/rageb/Desktop/new yolo model/datasets/valid_filtered/images")
VAL_LABEL_DIR  = Path("C:/Users/rageb/Desktop/new yolo model/datasets/valid_filtered/labels")
CLASS_NAMES    = ['car', 'emv', 'htv']
IOU_THRESH     = 0.5
CONF_THRESH    = 0.25
TB_LOG_DIR     = "runs/eval"
# ────────────────────────────────────────────────────────────────────────────────

num_classes = len(CLASS_NAMES)
bg_idx = num_classes - 1  # index of “background”

def xywh2xyxy(xc, yc, w, h, img_w, img_h):
    """Convert YOLO normalized xc,yc,w,h → absolute x1,y1,x2,y2."""
    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    return [x1, y1, x2, y2]

def compute_iou(b1, b2):
    """Compute IoU of two [x1,y1,x2,y2] boxes."""
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter_w, inter_h = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = inter_w * inter_h
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter/union if union>0 else 0

# ─── Load model ────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)

# ─── Prepare holders ───────────────────────────────────────────────────────────
y_true = []
y_pred = []

# ─── Loop over validation images ───────────────────────────────────────────────
for img_path in tqdm(list(VAL_IMG_DIR.rglob("*.jpg")), desc="Evaluating"):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # 1) Run inference
    results = model(img, conf=CONF_THRESH)[0]
    pred_boxes, pred_cls, pred_scores = [], [], []
    if results.boxes is not None:
        for box in results.boxes:
            pred_boxes.append(box.xyxy.cpu().numpy()[0].tolist())
            pred_cls.append(int(box.cls.cpu().numpy()[0]))
            pred_scores.append(float(box.conf.cpu().numpy()[0]))

    # 2) Load ground‑truth boxes
    gt_file = VAL_LABEL_DIR / f"{img_path.stem}.txt"
    if not gt_file.exists():
        continue

    gt_boxes, gt_cls = [], []
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                # skip malformed
                continue
            cid = int(parts[0])
            xc, yc, ww, hh = map(float, parts[1:5])
            gt_cls.append(cid)
            gt_boxes.append(xywh2xyxy(xc, yc, ww, hh, w, h))

    # 3) Match predictions → GT by descending score & IoU
    matched_gt   = set()
    matched_pred = set()
    # pair (pb,pc,ps,pidx)
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
            # matched pair
            y_true.append(gt_cls[best_i])
            y_pred.append(pc)
            matched_gt.add(best_i)
            matched_pred.add(pidx)

    # 4) Unmatched predictions → false positives (background → pred_class)
    for pidx, pc in enumerate(pred_cls):
        if pidx not in matched_pred:
            y_true.append(bg_idx)
            y_pred.append(pc)

    # 5) Unmatched GT → false negatives (gt_class → background)
    for gi, gc in enumerate(gt_cls):
        if gi not in matched_gt:
            y_true.append(gc)
            y_pred.append(bg_idx)

# ─── Build & Save Confusion Matrix───────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
cm_norm = cm.astype(float) / cm.sum(axis=1)[:,None]
disp = ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix")
plt.savefig("confusion_matrix_normalized.png")
plt.close()

# ─── Precision–Recall Curves (per class) ───────────────────────────────────────
y_true_bin = label_binarize(y_true, classes=range(num_classes))
y_pred_bin = label_binarize(y_pred, classes=range(num_classes))

plt.figure()
for i in range(num_classes):
    prec, rec, _ = precision_recall_curve(y_true_bin[:,i], y_pred_bin[:,i])
    pr_auc = auc(rec, prec)
    plt.plot(rec, prec, label=f"{CLASS_NAMES[i]} (AUC={pr_auc:.2f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curves")
plt.legend(loc="best")
plt.savefig("PR_curve.png")
plt.close()

# ─── Per‑class Precision / Recall / F1 ────────────────────────────────────────
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=range(num_classes)
)

plt.figure()
plt.plot(range(num_classes), prec, marker='o')
plt.xticks(range(num_classes), CLASS_NAMES)
plt.title("Precision per Class")
plt.savefig("P_curve.png")
plt.close()

plt.figure()
plt.plot(range(num_classes), rec, marker='o')
plt.xticks(range(num_classes), CLASS_NAMES)
plt.title("Recall per Class")
plt.savefig("R_curve.png")
plt.close()

plt.figure()
plt.plot(range(num_classes), f1, marker='o')
plt.xticks(range(num_classes), CLASS_NAMES)
plt.title("F1‑Score per Class")
plt.savefig("Fl_curve.png")
plt.close()

# ─── Label Correlation & Distribution ─────────────────────────────────────────
plt.figure()
plt.imshow(cm, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(num_classes), CLASS_NAMES)
plt.yticks(range(num_classes), CLASS_NAMES)
plt.title("Label Correlation Matrix")
plt.savefig("labels_correlogram.jpg")
plt.close()

plt.figure()
counts = np.bincount(y_true, minlength=num_classes)
plt.bar(CLASS_NAMES, counts)
plt.title("Label Distribution")
plt.savefig("labels.jpg")
plt.close()

# ─── TensorBoard Logging ──────────────────────────────────────────────────────
writer = SummaryWriter(TB_LOG_DIR)
writer.add_scalar("Eval/Mean_Precision", np.mean(prec), 0)
writer.add_scalar("Eval/Mean_Recall",    np.mean(rec),  0)
writer.add_scalar("Eval/Mean_F1",        np.mean(f1),   0)
writer.close()

print("✅ Done! All plots saved and TensorBoard logs in", TB_LOG_DIR)

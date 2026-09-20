# Object Detection Model Using YOLOv8

A YOLOv8-based vehicle detector for identifying `car`, `emv`, and `htv` in images and live webcam video. The repository includes training, inference, evaluation, visualizations, trained weights, and a Streamlit interface.

> **Dataset note:** `datasets/data.yaml` and the dataset image/label directories are referenced by the scripts but are not included in this repository. Provide the dataset or update the paths before training or evaluation.

## End-to-end flow

```mermaid
flowchart TD
    A[YOLO dataset<br/>images + labels + data.yaml] --> B[scripts/train.py]
    B --> C[YOLOv8 model<br/>models/yolov8_custom.yaml]
    C --> D[Training<br/>100 epochs, 640px, batch 16, CUDA]
    D --> E[best.pt checkpoint]

    E --> F{Inference}
    F --> G[Streamlit image upload]
    F --> H[Streamlit WebRTC webcam]
    F --> I[OpenCV image/webcam]
    G --> J[Preprocess image/frame]
    H --> J
    I --> J
    J --> K[YOLO forward pass]
    K --> L[Confidence filtering]
    L --> M[Boxes + class + confidence]
    M --> N[Annotated output]

    A --> O[Validation images and labels]
    E --> P[Ultralytics model.val]
    O --> P
    E --> Q[Custom metric scripts]
    O --> Q
    P --> R[mAP@0.5<br/>mAP@0.5:0.95]
    Q --> S[Precision, recall, F1,<br/>PR curves, confusion matrix]
```

## How YOLOv8 works

YOLO—**You Only Look Once**—is a single-stage object detector. Instead of first proposing regions and then classifying each region, it processes the image in one neural-network forward pass and directly predicts candidate boxes, objectness/class scores, and class labels.

This project uses a YOLOv8-style architecture with:

1. **Backbone:** convolutional and `C2f` blocks extract visual features at progressively smaller spatial resolutions.
2. **SPPF block:** spatial pyramid pooling aggregates context at multiple receptive-field sizes.
3. **Neck:** upsampling and concatenation fuse coarse semantic features with finer spatial features.
4. **Multi-scale detection head:** P3, P4, and P5 heads detect objects at different sizes, which is useful when vehicles are near or far from the camera.
5. **Post-processing:** confidence filtering and non-maximum suppression remove weak and duplicate detections before the boxes are returned.

The model resizes/letterboxes an input image, runs the forward pass, decodes predictions into pixel coordinates, applies post-processing, and returns bounding boxes, class IDs, and confidence values. In this repository, class IDs are mapped as `0=car`, `1=emv`, and `2=htv`.

## YOLOv8 compared with common alternatives

| Detector family | Main approach | Strengths | Trade-offs compared with YOLOv8 | Fit for this project |
|---|---|---|---|---|
| **YOLOv8** | One-stage, multi-scale detection | Strong speed/accuracy balance, simple Ultralytics API, easy training and deployment | Accuracy can depend heavily on data quality and threshold tuning | **Chosen:** supports both training and near-real-time webcam inference with little application code |
| **YOLOv5** | One-stage YOLO detector | Mature ecosystem, fast, widely deployed | Older architecture and tooling than YOLOv8; generally less convenient for newer experiments | A reasonable baseline, but YOLOv8 offers a newer API and architecture |
| **YOLOv7** | One-stage detector optimized for speed and accuracy | Very strong historical real-time performance | Less unified modern training/export workflow than Ultralytics YOLOv8 | Suitable for benchmarking, but adds migration and maintenance cost |
| **Faster R-CNN** | Two-stage region proposal plus classification | Often strong localization accuracy and useful for precision-focused workloads | Slower and more computationally expensive, especially for live video | Better when latency is less important than maximum localization quality |
| **SSD** | One-stage, multi-scale detector | Lightweight and comparatively simple | Often weaker accuracy, especially for small or crowded objects | Useful on constrained hardware, but less attractive for this vehicle detector |
| **RetinaNet** | One-stage detector with focal loss | Handles class imbalance well and can provide strong accuracy | Usually more complex/slower to deploy for this use case | Useful when imbalance is dominant, but YOLOv8 is simpler operationally |
| **DETR / RT-DETR** | Transformer-based set prediction | Global context and reduced reliance on traditional NMS in DETR-style designs; RT-DETR targets real-time use | Higher architectural complexity and potentially greater training/deployment cost | Worth evaluating when crowded scenes or global context become important |

These are general trade-offs rather than a universal ranking. A fair comparison requires the same dataset split, input size, hardware, augmentation policy, confidence/IoU settings, and evaluation procedure. YOLOv8 was selected here primarily for its practical speed, mature Python interface, multi-scale detection, and straightforward Streamlit/OpenCV integration.

## Repository layout

```text
app.py                         Streamlit image-upload and WebRTC webcam app
scripts/train.py               Training entry point
scripts/test.py                Ultralytics validation entry point
scripts/detect.py              OpenCV image/webcam inference
models/yolov8_custom.yaml      Three-class YOLOv8 architecture
metrics/plots.py               Metric and training-curve generation
plots/plot.py                  Extended evaluation utility
plots/results.csv              Archived per-epoch training history
runs/detect/train/weights/     best.pt and last.pt checkpoints
metrics/, plots/               Curves, confusion matrices, and sample images
wandb/                         Offline experiment-tracking artifacts
requirements.txt              Python dependencies
packages.txt                  Linux system packages
runtime.txt                   Python runtime declaration
```

## Environment setup

The declared runtime is Python 3.10. The dev container uses Python 3.11.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux, install the packages listed in `packages.txt`:

```bash
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
```

Important dependencies include Streamlit, Streamlit-WebRTC, Ultralytics `8.0.134`, PyTorch `2.2.0`, OpenCV, NumPy, and Pillow. Install a CUDA-compatible PyTorch build separately when GPU training is required.

## Dataset format

Create `datasets/data.yaml` in Ultralytics format:

```yaml
path: /absolute/path/to/datasets
train: train/images
val: valid/images
test: test/images
names:
  0: car
  1: emv
  2: htv
```

Every image needs a matching text file in its `labels` directory. Each label line contains normalized YOLO coordinates:

```text
<class_id> <center_x> <center_y> <width> <height>
```

## Training

`scripts/train.py` uses `models/yolov8_custom.yaml` and trains with 100 epochs, 640 × 640 images, batch size 16, CUDA, validation, automatic optimizer selection, and standard YOLO augmentations such as mosaic, horizontal flip, scale, HSV changes, and RandAugment defaults.

```bash
python scripts/train.py
```

Outputs are written to `runs/detect/train/`. The main checkpoints are:

- `best.pt`: checkpoint selected using validation performance; use this for inference.
- `last.pt`: checkpoint from the final epoch.

The script also exports a TorchScript model. The archived `runs/detect/train/args.yaml` records a previous run using `yolov8s.yaml` and pretrained weights, so a new run may not exactly reproduce the committed metrics.

## Inference

### Streamlit application

```bash
streamlit run app.py
```

The application loads `runs/detect/train/weights/best.pt`, accepts JPG/JPEG/PNG uploads, and provides a WebRTC webcam tab. The application uses a confidence threshold of `0.4` and caches the model with `@st.cache_resource`.

### OpenCV application

```bash
python scripts/detect.py
```

Choose `1` for webcam detection, `2` for image detection, or `q` to exit. On Linux/macOS, change the script's Windows-style checkpoint path to `runs/detect/train/weights/best.pt`.

### Programmatic inference

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("path/to/image.jpg", conf=0.4)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(class_id, confidence, (x1, y1, x2, y2))
```

## Metrics and evaluation

`scripts/test.py` calls `model.val(data="datasets/data.yaml")` and reports mAP values. The key metrics are:

- **Precision:** `TP / (TP + FP)`. Of all predicted objects, how many are correct. It measures false-alarm rate.
- **Recall:** `TP / (TP + FN)`. Of all ground-truth objects, how many were found. It measures missed detections.
- **IoU:** intersection area divided by union area between a predicted and ground-truth box.
- **AP:** area under a class-specific precision–recall curve as the confidence threshold varies.
- **mAP@0.5:** mean AP across classes using IoU ≥ 0.50.
- **mAP@0.5:0.95:** mean AP across IoU thresholds 0.50 through 0.95 in increments of 0.05. It is stricter and rewards tighter boxes.
- **F1:** `2 × precision × recall / (precision + recall)`, a balance between precision and recall.

A detection is a true positive when the class is correct and its box overlaps an unmatched ground-truth box above the selected IoU threshold. Unmatched predictions are false positives; unmatched labels are false negatives.

```bash
python scripts/test.py
```

> `scripts/test.py` references `runs\\detect\\train18\\weights\\best.pt`, but the committed checkpoint is in `runs/detect/train/weights/best.pt`. Update the path first.

### Archived results

The archived `plots/results.csv` contains 100 epochs. Its approximate results are:

- Best mAP@0.5: **0.8666** at epoch 42
- Best mAP@0.5:0.95: **0.6480** at epoch 67
- Epoch-100 precision: **0.8807**
- Epoch-100 recall: **0.7845**
- Epoch-100 mAP@0.5: **0.8486**
- Epoch-100 mAP@0.5:0.95: **0.6374**

These values describe the archived experiment and are not a guarantee for a retrained model.

### Custom plots

`metrics/plots.py` and `plots/plot.py` generate loss, learning-rate, precision, recall, F1, PR, and confusion-matrix plots. They use `conf=0.25`, convert YOLO normalized `xywh` labels to pixel `xyxy` boxes, sort predictions by confidence, and greedily match predictions to ground truth at IoU ≥ 0.5.

Update their hard-coded validation paths before running:

```bash
python metrics/plots.py
# or
python plots/plot.py
```

> **Caveat:** the custom scripts use `bg_idx = num_classes - 1`, which equals `2` and conflicts with the `htv` class. Use Ultralytics validation for formal reporting, or fix the scripts to use background ID `num_classes` and include that label in the confusion matrix.

## Limitations

- The dataset is not included, so training is not immediately reproducible from a clean clone.
- Several evaluation utilities contain absolute Windows paths.
- `scripts/test.py` uses a stale `train18` checkpoint path.
- Class names are duplicated across files; a shared configuration or `model.names` would reduce class-order drift.
- Confidence and IoU/NMS thresholds should be tuned on validation data according to whether false alarms or missed vehicles are more costly.

## License

No license file is included. Add an appropriate license before redistributing the code, model weights, or dataset-derived artifacts.

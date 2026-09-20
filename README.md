1| # Object Detection Model Using YOLOv8
2| 
3| A YOLOv8-based vehicle detector for identifying `car`, `emv`, and `htv` in images and live webcam video. The repository includes training, inference, evaluation, visualizations, trained weights, and a Streamlit interface for both upload and webcam workflows.
4| 
5| > **Dataset note:** `datasets/data.yaml` and the dataset image/label directories are referenced by the scripts but are not included in this repository. Provide the dataset or update the paths before training or validation.
6| 
7| ## End-to-end flow
8| 
9| ```mermaid
10| flowchart TD
11|     A[YOLO dataset: images, labels, data.yaml] --> B[scripts/train.py]
12|     B --> C[YOLOv8 model: models/yolov8_custom.yaml]
13|     C --> D[Training: 100 epochs, 640px, batch 16, CUDA]
14|     D --> E[best.pt checkpoint]
15| 
16|     E --> F{Inference}
17|     F --> G[Streamlit image upload]
18|     F --> H[Streamlit WebRTC webcam]
19|     F --> I[OpenCV image or webcam]
20|     G --> J[Preprocess image or frame]
21|     H --> J
22|     I --> J
23|     J --> K[YOLO forward pass]
24|     K --> L[Confidence filtering and NMS]
25|     L --> M[Boxes, class, confidence]
26|     M --> N[Annotated output]
27| 
28|     A --> O[Validation images and labels]
29|     E --> P[Ultralytics model.val]
30|     O --> P
31|     E --> Q[Custom metric scripts]
32|     O --> Q
33|     P --> R["mAP@0.5 and mAP@0.5:0.95"]
34|     Q --> S["Precision, recall, F1, PR curves, confusion matrix"]
35| ```
36| 
37| The model predicts bounding boxes, class IDs, and confidence values. Inference maps IDs to `0=car`, `1=emv`, and `2=htv`, filters detections, draws boxes with OpenCV, and displays the result.
38| 
39| ## How YOLOv8 works
40| 
41| YOLO—**You Only Look Once**—is a single-stage object detector. It processes an image in one neural-network forward pass and directly predicts candidate boxes, class scores, and confidence values.
42| 
43| This project uses a YOLOv8-style architecture with:
44| 
45| 1. **Input preprocessing:** images are resized/letterboxed to the configured size while preserving useful spatial proportions.
46| 2. **Backbone:** convolutional and `C2f` blocks extract features at progressively smaller spatial resolutions.
47| 3. **SPPF block:** spatial pyramid pooling gathers context at multiple receptive-field sizes.
48| 4. **Neck:** upsampling and concatenation fuse coarse semantic features with finer spatial features.
49| 5. **Multi-scale heads:** P3, P4, and P5 heads detect vehicles at different sizes, which helps with vehicles that are near or far from the camera.
50| 6. **Prediction decoding:** network outputs are converted into pixel-coordinate boxes, class IDs, and confidence scores.
| 7. **Post-processing:** confidence filtering and non-maximum suppression remove weak and duplicate detections.
52| 
53| For a detection to be useful, the predicted class must be correct and the predicted box must overlap the vehicle sufficiently. The confidence threshold controls the precision/recall trade-off: raising it reduces false positives but may miss small or distant objects.
54| 
55| ## YOLOv8 compared with alternatives
56| 
57| | Detector | Approach | Strengths | Trade-offs compared with YOLOv8 |
58| |---|---|---|---|
59| | **YOLOv8** | One-stage, multi-scale detector | Strong speed/accuracy balance, simple Ultralytics API, straightforward deployment | Accuracy depends on data quality and threshold tuning |
60| | **YOLOv5** | One-stage YOLO detector | Mature ecosystem, fast, widely deployed | Older architecture and tooling |
61| | **YOLOv7** | One-stage detector | Strong historical real-time performance | Less unified modern training/export workflow |
62| | **Faster R-CNN** | Two-stage region proposal and classification | Strong localization accuracy | Slower and more computationally expensive for live video |
63| | **SSD** | One-stage, multi-scale detector | Lightweight and simple | Often weaker accuracy, especially for small or crowded objects |
64| | **RetinaNet** | One-stage detector with focal loss | Handles class imbalance well | Usually more complex or slower to deploy for this use case |
65| | **DETR / RT-DETR** | Transformer-based detection | Global context; RT-DETR targets real-time use | Higher architectural complexity and potentially greater training cost |
66| 
67| These are general trade-offs, not universal rankings. A fair benchmark must use the same dataset split, input size, hardware, augmentation policy, confidence/IoU settings, and evaluation procedure.
68| 
69| ## Repository layout
70| 
71| ```text
72| app.py                         Streamlit image-upload and WebRTC webcam app
73| scripts/train.py               Training entry point
74| scripts/test.py                Ultralytics validation entry point
75| scripts/detect.py              OpenCV image/webcam inference
76| models/yolov8_custom.yaml      Three-class YOLOv8 architecture
77| metrics/plots.py               Metric and training-curve generation
78| plots/plot.py                  Extended evaluation utility
79| plots/results.csv              Archived per-epoch training history
80| runs/detect/train/weights/     best.pt and last.pt checkpoints
81| metrics/, plots/               Curves, confusion matrices, sample images
82| wandb/                         Offline experiment-tracking artifacts
83| requirements.txt              Python dependencies
84| packages.txt                   Linux system packages
85| runtime.txt                    Python runtime declaration
86| ```
87| 
88| ## Environment setup
89| 
90| The declared runtime is Python 3.10. Create an environment and install the dependencies:
91| 
92| ```bash
93| python -m venv .venv
94| source .venv/bin/activate       # Linux/macOS
95| # .venv\Scripts\Activate.ps1  # Windows PowerShell
96| pip install -r requirements.txt
97| ```
98| 
99| On Linux, install the packages listed in `packages.txt`:
100| 
101| ```bash
102| sudo apt-get update
103| sudo apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
104| ```
105| 
106| Important dependencies include Streamlit, Streamlit-WebRTC, Ultralytics `8.0.134`, PyTorch `2.2.0`, OpenCV, NumPy, and Pillow. Install a CUDA-compatible PyTorch build separately when GPU training is required.
107| 
108| ## Dataset format
109| 
110| Create `datasets/data.yaml` in Ultralytics format:
111| 
112| ```yaml
113| path: /absolute/path/to/datasets
114| train: train/images
115| val: valid/images
116| test: test/images
117| names:
118|   0: car
119|   1: emv
120|   2: htv
121| ```
122| 
123| Each image needs a matching text file in its `labels` directory. Each label line contains normalized YOLO coordinates:
124| 
125| ```text
126| <class_id> <center_x> <center_y> <width> <height>
127| ```
128| 
129| The coordinate values are normalized to `[0, 1]` relative to image width and height. `center_x` and `center_y` identify the box center; `width` and `height` identify its size.
130| 
131| ## Training
132| 
133| `scripts/train.py` uses `models/yolov8_custom.yaml` and trains with 100 epochs, 640 × 640 images, batch size 16, CUDA, validation, automatic optimizer selection, and standard augmentations such as mosaic and hue/saturation adjustments.
134| 
135| ```bash
136| python scripts/train.py
137| ```
138| 
139| Outputs are written to `runs/detect/train/`:
140| 
141| - `best.pt`: checkpoint selected using validation performance; use this for inference.
142| - `last.pt`: checkpoint from the final epoch.
143| - training history and plots generated by Ultralytics.
144| 
145| The script also exports a TorchScript model. The archived `runs/detect/train/args.yaml` records a previous run using `yolov8s.yaml` and pretrained weights, so a new run may not exactly reproduce the archived results.
146| 
147| ## Inference
148| 
149| ### Streamlit application
150| 
151| ```bash
152| streamlit run app.py
153| ```
154| 
155| The application loads `runs/detect/train/weights/best.pt`, accepts JPG/JPEG/PNG uploads, and provides a WebRTC webcam tab. It uses a confidence threshold of `0.4` and caches the model with `@st.cache_resource`.
156| 
157| ### OpenCV application
158| 
159| ```bash
160| python scripts/detect.py
161| ```
162| 
163| Choose `1` for webcam detection, `2` for image detection, or `q` to exit. On Linux/macOS, change the script's Windows-style checkpoint path to `runs/detect/train/weights/best.pt` if necessary.
164| 
165| ### Programmatic inference
166| 
167| ```python
168| from ultralytics import YOLO
169| 
170| model = YOLO("runs/detect/train/weights/best.pt")
171| results = model("path/to/image.jpg", conf=0.4)
172| 
173| for result in results:
174|     for box in result.boxes:
175|         class_id = int(box.cls.item())
176|         confidence = float(box.conf.item())
177|         x1, y1, x2, y2 = map(int, box.xyxy[0])
178|         print(class_id, confidence, (x1, y1, x2, y2))
179| ```
180| 
181| ## Metrics and evaluation
182| 
183| `scripts/test.py` calls `model.val(data="datasets/data.yaml")`. The key metrics are:
184| 
185| - **Precision:** `TP / (TP + FP)`, the fraction of predicted objects that are correct.
186| - **Recall:** `TP / (TP + FN)`, the fraction of ground-truth objects found.
187| - **IoU:** intersection area divided by union area between a predicted and ground-truth box.
188| - **AP:** area under a class-specific precision–recall curve as confidence varies.
189| - **mAP@0.5:** mean AP across classes using IoU ≥ 0.50.
| - **mAP@0.5:0.95:** mean AP across IoU thresholds 0.50 through 0.95 in increments of 0.05.
190| - **F1:** `2 × precision × recall / (precision + recall)`, balancing precision and recall.
191| 
192| A true positive has the correct class and sufficient IoU with an unmatched ground-truth box. Unmatched predictions are false positives; unmatched labels are false negatives.
193| 
194| ```bash
195| python scripts/test.py
196| ```
197| 
198| > `scripts/test.py` may reference `runs\\detect\\train18\\weights\\best.pt`, while the committed checkpoint is in `runs/detect/train/weights/best.pt`. Update the path before validation when necessary.
199| 
200| ### Archived results
201| 
202| The archived `plots/results.csv` contains 100 epochs:
203| 
204| - Best mAP@0.5: **0.8666** at epoch 42
205| - Best mAP@0.5:0.95: **0.6480** at epoch 67
206| - Epoch-100 precision: **0.8807**
207| - Epoch-100 recall: **0.7845**
208| - Epoch-100 mAP@0.5: **0.8486**
209| - Epoch-100 mAP@0.5:0.95: **0.6374**
210| 
211| These values describe the archived experiment and are not a guarantee for a retrained model.
212| 
213| ### Custom plots
214| 
215| `metrics/plots.py` and `plots/plot.py` generate loss, learning-rate, precision, recall, F1, PR, and confusion-matrix plots. They use `conf=0.25`, convert normalized YOLO `xywh` labels to pixel `xyxy`, and visualize validation outputs.
216| 
217| Update their hard-coded validation paths before running:
218| 
219| ```bash
220| python metrics/plots.py
221| # or
222| python plots/plot.py
223| ```
224| 
225| > **Caveat:** the custom scripts use `bg_idx = num_classes - 1`, which equals `2` and conflicts with the `htv` class. Use Ultralytics validation for formal reporting, or fix the scripts to use background indexing correctly.
226| 
227| ## Explanation of curves and metrics
228| 
229| ### Training and validation losses
230| 
231| Lower loss generally indicates that the model is making fewer training errors.
232| 
233| - **Box loss** measures how accurately predicted box coordinates match ground-truth boxes.
234| - **Class loss** measures whether detected objects are assigned the correct class.
235| - **DFL loss** means Distribution Focal Loss. YOLOv8 uses it to improve the precision of box edges.
236| - **Training loss** is calculated on images used to update model weights.
237| - **Validation loss** is calculated on unseen validation images. If training loss decreases while validation loss increases, the model may be overfitting.
238| 
239| In the archived run, training box loss decreases from about `3.33` to `0.51`, class loss from about `4.17` to `0.32`, and DFL loss from about `4.19` to `1.02` over 100 epochs.
240| 
241| ### Learning-rate curves
242| 
243| The `lr/pg0`, `lr/pg1`, and `lr/pg2` lines show learning rates for different optimizer parameter groups. The learning rate controls the size of weight updates and commonly decreases during training to stabilize convergence.
244| 
245| ### Precision, recall, and F1
246| 
247| Precision answers: **Of all objects predicted by the model, how many were correct?** High precision means relatively few false detections.
248| 
249| Recall answers: **Of all real objects in the images, how many did the model find?** High recall means relatively few missed vehicles.
250| 
251| F1 combines both:
252| 
253| ```text
254| F1 = 2 × (precision × recall) / (precision + recall)
255| ```
256| 
257| F1 is high only when precision and recall are both high. The F1 curve shows how this balance changes as the confidence threshold changes.
258| 
259| ### Precision-recall curve and PR AUC
260| 
261| The precision-recall curve plots recall against precision at different confidence thresholds. Raising the confidence threshold usually reduces false positives and increases precision, but can also lower recall because weaker detections are filtered out.
262| 
263| ### IoU and mAP
264| 
265| Intersection over Union measures overlap between a predicted box and its ground-truth box:
266| 
267| ```text
268| IoU = area of overlap / area of union
269| ```
270| 
271| An IoU of `0` means no overlap and `1.0` means a perfect overlap. **mAP@0.5** requires the correct class and IoU of at least `0.5`. **mAP@0.5:0.95** averages AP at IoU thresholds from `0.50` through `0.95`, usually in steps of `0.05`.
272| 
273| ### Normalized confusion matrix
274| 
275| The normalized confusion matrix compares true classes with predicted classes. Strong diagonal values (`car → car`, `emv → emv`, and `htv → htv`) indicate correct classification. Off-diagonal values show confusion between classes, such as a `car` being predicted as `htv` or vice versa.
276| 
277| > **Implementation note:** with three classes, `bg_idx = num_classes - 1` equals the `htv` class index. Background errors can therefore be mixed with `htv` in the custom matrix. Use `bg_idx = num_classes` or a dedicated background label to avoid ambiguity.
278| 
279| ### Training batches and validation examples
280| 
281| - **Training-batch images** verify that images and annotations load correctly and that augmentation preserves the labels.
282| - **Validation-label images** show the ground-truth annotations used during evaluation.
283| - **Validation-prediction images** show predicted boxes, classes, and confidence values and help identify missed vehicles, false positives, class confusion, and inaccurate box placement.
284| 
285| ## Limitations
286| 
287| - The dataset is not included, so training is not immediately reproducible from a clean clone.
288| - Several evaluation utilities contain absolute Windows paths; update them before running.
289| - `scripts/test.py` may reference a stale checkpoint path; use `runs/detect/train/weights/best.pt` when necessary.
290| - Class names are duplicated across files; a shared configuration or `model.names` would reduce class-order drift.
291| - Confidence and IoU/NMS thresholds should be tuned on validation data according to whether false alarms or missed vehicles are more costly.
292| - Archived metric values apply only to the archived experiment and may differ after retraining.
293| 
294| ## Generated evaluation plots
295| 
296| The repository already contains the generated training and evaluation charts. These plots are useful for quickly checking whether the model is learning correctly and whether the validation metrics are improving over time.
297| 
298| ### Training curves
299| 
300| ![Loss curves](plots/losses_plot.png)
301| 
302| ![Learning rate curves](plots/learning_rates_plot.png)
303| 
304| ![Metrics plot](plots/metrics_plot.png)
305| 
306| ### Precision and recall curves
307| 
308| ![Precision curve](plots/P_curve.png)
309| 
310| ![Recall curve](plots/R_curve.png)
311| 
312| ![F1 curve](plots/F1_curve.png)
313| 
314| ![Precision-recall curve](plots/PR_curve.png)
315| 
316| ### Confusion matrix and validation outputs
317| 
318| ![Normalized confusion matrix](plots/confusion_matrix_normalized.png)
319| 
320| ![Validation batch 0 predictions](plots/val_batch0_pred.jpg)
321| 
322| ![Validation batch 1 predictions](plots/val_batch1_pred.jpg)
323| 
324| ![Validation batch 2 predictions](plots/val_batch2_pred.jpg)
325| 
326| ## Metric calculation summary
327| 
328| The model is evaluated by comparing predictions to ground-truth bounding boxes and labels.
329| 
330| - **True positive (TP):** a detection matches the right class and has enough overlap with the correct object.
331| - **False positive (FP):** a prediction is made but it does not match a real object, or the duplicate detection is counted as extra.
332| - **False negative (FN):** a true object is missed by the detector.
333| 
334| ```text
335| Precision = TP / (TP + FP)
336| Recall    = TP / (TP + FN)
337| F1        = 2 × Precision × Recall / (Precision + Recall)
338| ```
339| 
340| The IoU score is calculated as:
341| 
342| ```text
343| IoU = area of overlap / area of union
344| ```
345| 
346| A prediction is counted as a correct match only when the class is correct and the IoU is above the chosen threshold. For example, with `mAP@0.5`, a detection must have IoU ≥ 0.5 and the correct class to count toward the AP score. For `mAP@0.5:0.95`, the model is evaluated at multiple IoU thresholds and the average result is reported.
347| 
348| Average precision (AP) is computed from the precision-recall curve for each class. The mean average precision (mAP) is the average AP across classes, and is what is used as the main detection-quality metric in this project.
349| 
350| ## License
351| 
352| No license file is included. Add an appropriate license before redistributing the code, model weights, or dataset-derived artifacts.
353| 

import cv2
import torch
from ultralytics import YOLO

# Set the model path
model_path = r"runs\detect\train\weights\best.pt"

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Try loading the YOLO model
try:
    model = YOLO(model_path)
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()

# Set higher confidence threshold (you can tune this)
model.conf = 0.4

# Class names (optional, if not using model.names)
# Assuming 0 = car, 1 = emv, 2 = htv, 3 = background
class_names = ['car', 'emv', 'htv']

# Global variable to stop live detection when clicked
stop_live_detection = False

def click_event(event, x, y, flags, param):
    global stop_live_detection
    if event == cv2.EVENT_LBUTTONDOWN:
        stop_live_detection = True

while True:
    mode = input("\nEnter '1' for live detection, '2' for image detection, or 'q' to quit: ").strip().lower()

    if mode == "1":
        cap = cv2.VideoCapture(0)
        stop_live_detection = False

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            continue

        cv2.namedWindow("YOLOv8 Real-Time Detection")
        cv2.setMouseCallback("YOLOv8 Real-Time Detection", click_event)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_live_detection:
                break

            results = model(frame)[0]

            for box in results.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())

                # Skip background class (id 3)
                if cls_id == 3:
                    continue

                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Real-Time Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Live detection stopped.")

    elif mode == "2":
        image_path = input("Enter the path of the image: ").strip()

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Error: Image file not found at {image_path}")

            results = model(image)[0]

            for box in results.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())

                if cls_id == 3:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_names[cls_id]} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow("YOLOv8 Image Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error: {e}")

    elif mode == "q":
        print("Exiting program.")
        break

    else:
        print("Invalid input. Please enter '1' for real-time, '2' for image detection, or 'q' to quit.")

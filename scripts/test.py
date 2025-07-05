from ultralytics import YOLO

def test_model():
    # Load the trained YOLO model with the correct path
    model = YOLO(r"runs\detect\train18\weights\best.pt")

    # Evaluate the model on the test set
    metrics = model.val(data=r"datasets\data.yaml")

    # Print key detection metrics
    print(f"mAP@50: {metrics.box.map50:.4f}")  # Mean Average Precision at IoU=0.5
    print(f"mAP@50-95: {metrics.box.map:.4f}")  # Mean Average Precision averaged over IoU=0.5:0.95

if __name__ == "__main__":
    test_model()

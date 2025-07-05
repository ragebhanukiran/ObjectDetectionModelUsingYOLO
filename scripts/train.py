
from ultralytics import YOLO

if __name__ == "__main__":
    # Load YOLO model
    model = YOLO("yolov8s.yaml")

    # Train the model
    results = model.train(
        data="datasets/data.yaml",
        epochs=100,
        imgsz=640,
        workers=0,
        batch=16,
        device="cuda",
        plots = False
    )
    # Save model
    model_path = "runs/detect/train/weights/best.pt"
    model.export(format="torchscript")
    print(f" Model saved at: {model_path}")

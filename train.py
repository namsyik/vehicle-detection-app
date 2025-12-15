from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=100,        # increased epochs
        imgsz=640,
        batch=16,
        device="cpu",
        patience=20        # early stopping
    )

if __name__ == "__main__":
    main()

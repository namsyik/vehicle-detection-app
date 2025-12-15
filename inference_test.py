from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")

results = model(
    "dataset/test/images",
    conf=0.4,
    save=True
)

for r in results:
    print(r.names)

print("Check runs/detect/predict/")

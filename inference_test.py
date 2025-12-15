from ultralytics import YOLO

model = YOLO("best.pt")

results = model(
    "dataset/test/images",
    conf=0.4,
    save=True
)

for r in results:
    print(r.names)

print("Inference tested successfully")

from ultralytics import YOLO

# Load the YOLOv8 nano model (fastest)
model = YOLO("yolov8n.pt")

from ultralytics import YOLO

# Load smallest model for fastest training
model = YOLO("yolov8n.pt")

# Train with aggressive learning rate
model.train(
    data="dataset.yaml",       # Path to your dataset YAML
    epochs=50,                  # very few epochs
    imgsz=416,                  # smaller size for speed
    batch=8,                    # low batch for CPU
    device="cpu",               # force CPU
    workers=0,                  # avoid Windows multiprocessing
    optimizer="Adam",           # faster convergence for small datasets
    lr0=0.01,                    # higher initial learning rate
    lrf=0.1,                     # final learning rate fraction
    project="C:/Users/ASUS/Documents/Cashlift/models",  # save location
    name="cashlift_train"         # subfolder name
)

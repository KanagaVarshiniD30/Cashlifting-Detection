from ultralytics import YOLO

# Load the YOLOv8 nano model (fastest)
model = YOLO("yolov8n.pt")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset.yaml",       
    epochs=50,                  
    imgsz=416,                  
    batch=12,                    
    device="cpu",               # force CPU
    workers=0,                  # avoid Windows multiprocessing
    optimizer="Adam",           
    lr0=0.01,                    # higher initial learning rate
    lrf=0.1,                     
    project="C:/Users/ASUS/Documents/Cashlift/models",  # save location
    name="cashlift_train"         # subfolder name
)


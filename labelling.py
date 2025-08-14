import os
import cv2
from ultralytics import YOLO

# Paths
base_dirs = [
    r"C:/Users/ASUS/Documents/Cashlift/processed_frames/processed_normal",
    r"C:/Users/ASUS/Documents/Cashlift/processed_frames/processed_cashlift"
]
output_labels_dir = r"C:/Users/ASUS/Documents/Cashlift/auto_labels"

os.makedirs(output_labels_dir, exist_ok=True)

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # COCO-pretrained

# Map COCO -> custom classes
# COCO class 0 = person → cashier (0)
# No direct COCO hand class, we will just use 'person' for now, refine later
coco_to_custom = {0: 0}

# Loop through both folders
for folder in base_dirs:
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder, filename)
            results = model.predict(img_path, conf=0.5, verbose=False)

            # Open image to get width/height
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            label_lines = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in coco_to_custom:
                        custom_id = coco_to_custom[cls_id]
                        x1, y1, x2, y2 = box.xyxy[0]
                        # Convert to YOLO format (x_center, y_center, width, height) normalized
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        label_lines.append(f"{custom_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Save label file
            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(output_labels_dir, label_file)
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

            print(f"[INFO] Processed {filename} → {len(label_lines)} boxes")

print(f"[DONE] Labels saved to: {output_labels_dir}")

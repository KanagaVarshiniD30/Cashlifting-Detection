import cv2
import os
from ultralytics import YOLO
from collections import deque
import time
import datetime

# ==== SETTINGS ====
MODEL_PATH = r"C:/Users/ASUS/Documents/Cashlift/runs/detect/cashlifting_fast2/weights/best.pt"
SAVE_ALERTS_TO = r"C:/Users/ASUS/Documents/Cashlift/alerts"
VIDEO_SOURCE = r"C:/Users/ASUS/Documents/Cashlift/dataset2/cashlift/cctv_CAM 5_main_20250524212319.dav"
CONFIDENCE = 0.5
FPS = 30  # change if your video FPS is different

# Create alerts folder
os.makedirs(SAVE_ALERTS_TO, exist_ok=True)

# ==== Load model ====
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

# ==== Circular buffer for last 10 seconds ====
buffer_size = FPS * 10
frame_buffer = deque(maxlen=buffer_size)

# ==== Video capture ====
cap = cv2.VideoCapture(VIDEO_SOURCE)

cashier_zone = None
alert_active = False
alert_clip_count = 0

print("[INFO] Starting Cashlifting Detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_buffer.append(frame.copy())  # save to buffer

    # Predict
    results = model.predict(frame, conf=CONFIDENCE, verbose=False)

    current_cashier_box = None
    hand_boxes = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = CLASS_NAMES[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            color = (0, 255, 0) if label == "cashier" else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if label == "cashier":
                current_cashier_box = (x1, y1, x2, y2)
            elif label == "hand":
                hand_boxes.append((x1, y1, x2, y2))

    # Set cashier zone only once
    if cashier_zone is None and current_cashier_box:
        cashier_zone = current_cashier_box

    # Draw highlighted cashier zone
    if cashier_zone:
        cx1, cy1, cx2, cy2 = cashier_zone

        # Transparent yellow overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (0, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        # Yellow border
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 3)
        cv2.putText(frame, "Cashier Zone", (cx1, cy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Detect if hand enters cashier zone
        for hx1, hy1, hx2, hy2 in hand_boxes:
            if hx1 > cx1 and hy1 > cy1 and hx2 < cx2 and hy2 < cy2:
                if not alert_active:
                    alert_active = True
                    alert_clip_count += 1
                    print("[ALERT] Possible cashlifting detected!")

                cv2.putText(frame, "ALERT: Possible Cashlifting!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Save alert video
    if alert_active:
        filename = os.path.join(
            SAVE_ALERTS_TO,
            f"alert_{alert_clip_count}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        height, width = frame.shape[:2]
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
        for f in list(frame_buffer):
            out.write(f)
        out.release()
        print(f"[SAVED] Alert video saved: {filename}")
        alert_active = False  # reset

    # Display feed
    cv2.imshow("Cashlifting Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

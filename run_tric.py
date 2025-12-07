import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import time

# ============================================================
# CONFIG
# ============================================================

# Path to your trained tricycle body number model
MODEL_PATH = r"C:\Users\USER\Documents\PHRoads\TricycleBodyNumberC\runs\detect\tricycle_body3\weights\best.pt"

# Video source:
#   0              -> webcam
#   "test.mp4"     -> local video file
#   "rtmp://... "  -> RTMP stream
SOURCE = r"C:\Users\USER\Documents\PHRoads\testt.mp4"

# Confidence threshold for YOLO detections
CONF_THRESH = 0.4

# Use GPU for EasyOCR if available (True/False)
EASYOCR_GPU = True

# 🔢 Body number format: 2–4 digits only
MIN_DIGITS = 2
MAX_DIGITS = 4


# ============================================================
# INIT
# ============================================================

print("🚦 Running tricycle body number detection + OCR... Press 'Q' to quit.")

# Load YOLO model
model = YOLO(MODEL_PATH)
class_names = model.names  # dict: {0: 'tricycle', 1: 'tricycle-body-number', ...}

# Init EasyOCR (digits only via allowlist later)
reader = easyocr.Reader(['en'], gpu=EASYOCR_GPU)

# Open video source
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print(f"❌ Failed to open video source: {SOURCE}")
    raise SystemExit

# Optional: get frame size (used if you later want to save a video)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  No more frames (or failed to read frame). Exiting.")
        break

    results = model(frame)[0]

    if results.boxes is not None:
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue

            cls_id = int(box.cls[0])
            cls_name = class_names.get(cls_id, str(cls_id))

            if cls_name != "tricycle-body-number":
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

            crop = frame[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            if h < 10 or w < 10:
                continue

            crop_up = cv2.resize(
                crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            )

            ocr_results = reader.readtext(
                crop_up,
                detail=1,
                paragraph=False,
                allowlist="0123456789",
            )

            body_number = ""
            best_conf = 0.0

            for (bbox, text, ocr_conf) in ocr_results:
                # Keep digits only
                digits_only = "".join(ch for ch in text if ch.isdigit())
                if not digits_only:
                    continue

                # Enforce 2–4 digits
                if len(digits_only) < MIN_DIGITS:
                    continue
                if len(digits_only) > MAX_DIGITS:
                    # e.g. "012345" -> "2345" (keep last MAX_DIGITS)
                    digits_only = digits_only[-MAX_DIGITS:]

                if ocr_conf > best_conf:
                    best_conf = ocr_conf
                    body_number = digits_only

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if body_number:
                label = f"{body_number} ({best_conf:.2f})"
            else:
                label = "body-num (?)"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.imshow("Tricycle Body Number Detection + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
print("✅ Finished.")

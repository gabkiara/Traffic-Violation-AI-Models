import cv2
import time
import re
import numpy as np
import easyocr
from ultralytics import YOLO

# === Load YOLOv8 Plate Detection model ===
model = YOLO(
    r"C:\Users\USER\Documents\PHRoads\PlateDetectionE\runs\detect\train\weights\best.pt"
)

# === Video source (file or RTSP) ===
video_path = r"C:/Users/USER/Documents/PHRoads/testvid.mp4"
cap = cv2.VideoCapture(video_path)

# === Initialize EasyOCR ===
ocr_reader = easyocr.Reader(["en"])

# === Output video writer ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None
if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(
        "output_plate_detection.mp4", fourcc, fps_video, (width, height)
    )

# --------------------------------------------------------------------
#              PLATE TEXT NORMALIZATION & FORMAT CHECK
# --------------------------------------------------------------------


def normalize_plate_text(text: str) -> str:
    """
    Normalize OCR result:
    - Uppercase
    - Keep only letters/numbers
    - Collapse multiple spaces
    """
    if not text:
        return ""

    text = text.upper()
    # Replace non-alphanumeric with space
    text = re.sub(r"[^A-Z0-9]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Car: ABC 1234
CAR_PATTERNS = [
    re.compile(r"^[A-Z]{3}\s?\d{4}$"),
]

# Motorcycle patterns:
MOTOR_PATTERNS = [
    re.compile(r"^\d{3}\s?[A-Z]{3}$"),        # 123 ABC
    re.compile(r"^[A-Z]\s?\d{3}\s?[A-Z]{2}$"),  # A 123 BC
    re.compile(r"^[A-Z]{2}\s?\d{3}\s?[A-Z]$")  # AB 123 C
]


def classify_plate_format(norm_text: str) -> str:
    """
    Return 'CAR', 'MOTOR', or 'UNKNOWN' based on normalized plate text.
    """
    if not norm_text:
        return "UNKNOWN"

    for p in CAR_PATTERNS:
        if p.match(norm_text):
            return "CAR"

    for p in MOTOR_PATTERNS:
        if p.match(norm_text):
            return "MOTOR"

    return "UNKNOWN"


# === License Plate Detection and OCR helper ===
def get_plate_text(cropped_image) -> str:
    """Detect text from the cropped license plate using EasyOCR."""
    if cropped_image is None or cropped_image.size == 0:
        return ""

    result = ocr_reader.readtext(cropped_image)
    text = " ".join([entry[1] for entry in result])  # entry[1] is the text
    return text.strip()


# === Plate detection and OCR processing ===
prev_time = time.time()
fps = 0.0

print("🚦 Running plate detection and OCR... Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === Calculate FPS ===
    curr_time = time.time()
    fps = 1.0 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    # === Run YOLOv8 detection for plates ===
    results = model(frame, conf=0.6, verbose=False)
    dets = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < 0.6:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        dets.append((x1, y1, x2, y2, conf, label))

    # === Draw bounding boxes and extract plates for OCR ===
    for x1, y1, x2, y2, conf, label in dets:
        # Crop plate region
        crop = frame[
            max(0, int(y1)) : min(int(y2), frame.shape[0]),
            max(0, int(x1)) : min(int(x2), frame.shape[1]),
        ]

        # OCR
        raw_text = get_plate_text(crop)
        norm_text = normalize_plate_text(raw_text)
        plate_type = classify_plate_format(norm_text)

        # Draw box
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

        # Use color by type: CAR=green, MOTOR=blue, UNKNOWN=yellow
        if plate_type == "CAR":
            color = (0, 255, 0)
        elif plate_type == "MOTOR":
            color = (255, 0, 0)
        else:
            color = (0, 255, 255)

        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)

        # HUD text: TYPE: PLATE
        display_text = norm_text if norm_text else "N/A"
        hud = f"{plate_type}: {display_text}"
        cv2.putText(
            frame,
            hud,
            (x1i, max(y1i - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # === Display FPS ===
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

    cv2.imshow("Plate Detection and OCR", frame)

    if out:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
print("✅ Processing complete. Results saved to output_plate_detection.mp4")

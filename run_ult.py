# pip install ultralytics deep-sort-realtime easyocr opencv-python-headless torch torchvision

import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr

# Load all YOLO models
vehicle_model   = YOLO("/VehicleDetection/runs/detect/vehicle_detection_finetune/weights/best.pt")
plate_model     = YOLO("/PlateDetection/runs/detect/plate_detection/weights/best.pt")
tricycle_model  = YOLO("/TricycleBodyNumberDetection/runs/detect/tricycle_body_number_model/weights/best.pt")
violation_model = YOLO("/ViolationDetection/runs/detect/violation_detection_model/weights/best.pt")

# Load EasyOCR for plate/body-number reading
ocr_reader = easyocr.Reader(['en'])

# Tracker (shared for vehicles)
tracker = DeepSort(max_age=50, n_init=5, nn_budget=100)

# Video source (your RTSP)
video_path = "rtsp://username:password@<camera-ip>:554/stream"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === Step 1: Vehicle detection + tracking ===
    veh_results = vehicle_model(frame, conf=0.6, verbose=False)
    dets = []
    for box in veh_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = vehicle_model.names[int(box.cls[0])]
        dets.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    tracks = tracker.update_tracks(dets, frame=frame)
    for t in tracks:
        if not t.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        crop = frame[y1:y2, x1:x2]

        # === Step 2: Detect plates inside vehicles ===
        plate_results = plate_model(crop, conf=0.6, verbose=False)
        for pb in plate_results[0].boxes:
            px1, py1, px2, py2 = map(int, pb.xyxy[0])
            plate_crop = crop[py1:py2, px1:px2]
            text = ocr_reader.readtext(plate_crop)
            if text:
                plate_text = text[0][1]
                cv2.putText(frame, f"Plate: {plate_text}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # === Step 3: Tricycle body-number OCR ===
        trike_results = tricycle_model(crop, conf=0.6, verbose=False)
        for tb in trike_results[0].boxes:
            bx1, by1, bx2, by2 = map(int, tb.xyxy[0])
            body_crop = crop[by1:by2, bx1:bx2]
            body_text = ocr_reader.readtext(body_crop)
            if body_text:
                num_text = body_text[0][1]
                cv2.putText(frame, f"Body #: {num_text}", (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # === Step 4: Violation detection ===
        viol_results = violation_model(crop, conf=0.6, verbose=False)
        people = 0
        helmets = 0
        for vb in viol_results[0].boxes:
            lbl = violation_model.names[int(vb.cls[0])]
            if lbl == "person":
                people += 1
            elif lbl == "helmet":
                helmets += 1

        if people > 2:
            cv2.putText(frame, "Violation: Overload", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif people > 0 and helmets < people:
            cv2.putText(frame, "Violation: No Helmet", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # === Display / Stream Output ===
    cv2.imshow("Unified AI Traffic Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

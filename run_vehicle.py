from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model
model = YOLO(r"C:\Users\USER\Documents\PHRoads\VehicleDetectionL\runs\detect\train3\weights\best.pt")

video_path =r"C:/Users/USER/Documents/PHRoads/testtt.mp4"

cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None
if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("output_detection_test.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection only (not tracking)
    results = model(frame)

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf >= 0.7:  
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            text = f"{label} {conf:.2f}"  # keep confidence

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection", frame)

    if out:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

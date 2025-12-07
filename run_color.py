from ultralytics import YOLO
import cv2
import numpy as np
import time

def main():
    # Load the trained color detection model
    model = YOLO(r"C:\Users\USER\Documents\PHRoads\ColorDetectionE\runs\detect\train\weights\best.pt")

    # Open the video source (local video or RTSP stream)
    video_path ="C:\\Users\\USER\\Documents\\PHRoads\\vidtest.mp4"
    cap = cv2.VideoCapture(video_path)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter("output_color_detectionE.mp4", fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection for color detection
        results = model(frame)

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf >= 0.6:  # Filter out low-confidence detections
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} (Color Detected)"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Write the frame with color annotations to the output video
        if out:
            out.write(frame)

        # Display the resulting frame with color labels
        cv2.imshow("Color Detection", frame)

        # Press 'Q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("✅ Processing complete! Results saved to output_color_detection.mp4")

if __name__ == "__main__":
    main()

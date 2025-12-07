import cv2
import time
from ultralytics import YOLO


def main():
    # Load the trained YOLO model
    model = YOLO(
        r"C:\Users\USER\Documents\PHRoads\ViolationDetectionD\runs\detect\train\weights\best.pt"
    )

    # Open the video file or RTSP stream
    video_path = r"C:/Users/USER/Documents/PHRoads/testt.mp4"
    cap = cv2.VideoCapture(video_path)

    # FPS and writer
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(
        "output_violation.mp4",
        fourcc,
        fps,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        people_count = 0
        helmet_count = 0
        no_helmet_count = 0
        violation_count = 0

        # Store boxes for association
        motorcycles = []  # list of dicts: {"bbox": (x1,y1,x2,y2)}
        persons = []      # list of dicts: {"bbox": (x1,y1,x2,y2)}

        # ---- 1. First pass: draw basic boxes, collect data ----
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == "motorcycle":
                motorcycles.append({"bbox": (x1, y1, x2, y2)})
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Motorcycle",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            elif label == "person":
                people_count += 1
                persons.append({"bbox": (x1, y1, x2, y2)})
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    "Person",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

            elif label == "helmet":
                helmet_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "Helmet",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            elif label in ["no-helmet", "no_helmet"]:
                no_helmet_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "No Helmet",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        # ---- 2. Associate persons to motorcycles (for overloading) ----
        moto_riders = [[] for _ in motorcycles]  # indices of persons per motorcycle

        for p_idx, p in enumerate(persons):
            px1, py1, px2, py2 = p["bbox"]
            cx = (px1 + px2) / 2.0
            cy = (py1 + py2) / 2.0

            # find motorcycle whose expanded box contains this person center
            for m_idx, m in enumerate(motorcycles):
                mx1, my1, mx2, my2 = m["bbox"]

                # expand motorcycle box a bit to cover riders
                ex1 = mx1 - 30
                ey1 = my1 - 60
                ex2 = mx2 + 30
                ey2 = my2 + 40

                if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                    moto_riders[m_idx].append(p_idx)
                    break  # assign to first matching motorcycle

        # ---- 3. Draw overloading bounding boxes ----
        overloaded_bikes = 0
        for m_idx, riders in enumerate(moto_riders):
            if len(riders) > 2:  # >2 persons assigned to the same motorcycle
                overloaded_bikes += 1
                mx1, my1, mx2, my2 = motorcycles[m_idx]["bbox"]

                # big red box around the overloaded motorcycle
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 4)
                cv2.putText(
                    frame,
                    f"OVERLOAD ({len(riders)} riders)",
                    (mx1, max(my1 - 15, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        if overloaded_bikes > 0:
            violation_count += overloaded_bikes
            cv2.putText(
                frame,
                f"🚨 Overloading: {overloaded_bikes} motorcycle(s)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # ---- 4. No-helmet violations (still text-based) ----
        if no_helmet_count > 0:
            violation_count += 1
            cv2.putText(
                frame,
                "🚨 Violation: No Helmet",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        elif people_count > 0 and helmet_count < people_count:
            # fallback logic if you still want it:
            violation_count += 1
            cv2.putText(
                frame,
                "🚨 Possible No Helmet",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # ---- 5. HUD text ----
        cv2.putText(
            frame,
            f"People: {people_count}  Helmets: {helmet_count}  No-Helmet: {no_helmet_count}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Violations: {violation_count}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Violation Detection", frame)
        out.write(frame)

        # keep playback roughly real-time
        time.sleep(1.0 / fps)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Processing complete!")


if __name__ == "__main__":
    main()

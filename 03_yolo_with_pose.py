import cv2
from ultralytics import YOLO
import mediapipe as mp

def main(webcam_id=0, conf=0.4):
    # YOLO model
    yolo = YOLO("yolov8n.pt")

    # MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(webcam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Webcam tidak terdeteksi.")
        return

    print("Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1. YOLO Detection
        yolo_results = yolo(frame)[0]

        for box in yolo_results.boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue  # hanya proses 'person'

            # koordinat bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            # 2. Pose Estimator pada crop
            crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(crop_rgb)

            # 3. Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            if pose_result.pose_landmarks:
                # 4. Gambar landmark ke posisi asli frame
                for lm in pose_result.pose_landmarks.landmark:
                    lx = int(lm.x * (x2 - x1)) + x1
                    ly = int(lm.y * (y2 - y1)) + y1
                    cv2.circle(frame, (lx, ly), 3, (0,0,255), -1)

                # 5. Gambar connections (tulang)
                mp_drawing.draw_landmarks(
                    frame[y1:y2, x1:x2],
                    pose_result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

        cv2.imshow("YOLO + Pose Estimator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

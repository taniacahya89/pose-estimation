from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import joblib
import argparse
import os

NUM_LANDMARKS = 33

def normalize_row_flat(flat):
    arr = np.array(flat).reshape(NUM_LANDMARKS, 4)
    xs = arr[:,0].astype(float)
    ys = arr[:,1].astype(float)
    # hip center
    try:
        hip = np.nanmean(arr[[23,24], :2], axis=0)
    except:
        hip = np.nanmean(arr[:, :2], axis=0)
    xs = xs - hip[0]
    ys = ys - hip[1]
    try:
        shoulders = np.nanmean(arr[[11,12], :2], axis=0)
        torso_dist = np.linalg.norm(shoulders - hip)
        if torso_dist == 0 or np.isnan(torso_dist):
            torso_dist = 1.0
    except:
        torso_dist = 1.0
    xs = xs / torso_dist
    ys = ys / torso_dist
    feat = np.concatenate([xs, ys])
    return feat

def main(model_path='models/model_pose.pkl', webcam_id=0, conf=0.4):
    if not os.path.exists(model_path):
        print(f"Model tidak ditemukan: {model_path}. Jalankan 05_train_classifier.py dulu.")
        return

    data = joblib.load(model_path)
    clf = data['model']
    scaler = data['scaler']
    le = data['label_encoder']

    yolo = YOLO("yolov8n.pt")
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(webcam_id)
    if not cap.isOpened():
        print("Webcam tidak ditemukan.")
        return

    print("Realtime pose classification. Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame)
        res = results[0]
        boxes = []
        if hasattr(res, 'boxes') and len(res.boxes) > 0:
            for box in res.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    boxes.append((x1,y1,x2,y2))

        disp = frame.copy()
        for (x1,y1,x2,y2) in boxes:
            x1c, y1c = max(x1,0), max(y1,0)
            x2c, y2c = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res_pose = pose.process(rgb)
            if not res_pose.pose_landmarks:
                # tampilkan box tanpa label
                cv2.rectangle(disp, (x1c,y1c), (x2c,y2c), (0,255,0), 1)
                continue

            lm = res_pose.pose_landmarks.landmark
            flat = []
            for i in range(NUM_LANDMARKS):
                flat += [lm[i].x, lm[i].y, lm[i].z, lm[i].visibility]
            feat = normalize_row_flat(flat)
            feat_s = scaler.transform([feat])
            pred_idx = clf.predict(feat_s)[0]
            label = le.inverse_transform([pred_idx])[0]

            # overlay skeleton on crop (optional)
            mp_draw.draw_landmarks(crop, res_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            disp[y1c:y2c, x1c:x2c] = crop

            # overlay box + label
            cv2.rectangle(disp, (x1c,y1c), (x2c,y2c), (0,255,0), 2)
            cv2.putText(disp, f"{label}", (x1c, y1c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Realtime Pose Classification", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/model_pose.pkl')
    parser.add_argument('--webcam', type=int, default=0)
    args = parser.parse_args()
    main(model_path=args.model, webcam_id=args.webcam)

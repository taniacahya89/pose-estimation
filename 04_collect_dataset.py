import argparse
import os
import time
import csv
import traceback
from ultralytics import YOLO
import cv2
import mediapipe as mp

NUM_LANDMARKS = 33  # MediaPipe Pose

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def build_header():
    header = []
    for i in range(NUM_LANDMARKS):
        header += [f"x{i}", f"y{i}", f"z{i}", f"vis{i}"]
    header += ["label", "timestamp"]
    return header

def open_webcam_with_debug(preferred_id):
    """Membuka webcam dengan pola debug seperti script sebelumnya."""

    print("\n=== Webcam Debug Opening ===")
    test_ids = [preferred_id, 0, 1, 2, 700, 1400]
    checked = set()

    for wid in test_ids:
        if wid in checked:
            continue
        checked.add(wid)

        print(f"Mencoba webcam id={wid} ...")
        cap = cv2.VideoCapture(wid, cv2.CAP_DSHOW)
        print("  cap.isOpened():", cap.isOpened())

        if cap.isOpened():
            print(f"==> Webcam terbuka dengan ID {wid}\n")
            return cap

        cap.release()

    print("Tidak ada webcam yang bisa dibuka.\n")
    return None


def main():
    parser = argparse.ArgumentParser(description="Collect pose keypoints to CSV using YOLO+MediaPipe")
    parser.add_argument('label', type=str, help='Nama label (mis. standing, typing)')
    parser.add_argument('--duration', type=int, default=30, help='Lama rekam (detik)')
    parser.add_argument('--webcam', type=int, default=0, help='ID webcam (cek test_cam.py kalau ragu)')
    parser.add_argument('--outdir', type=str, default='dataset')
    parser.add_argument('--conf', type=float, default=0.4, help='YOLO confidence threshold')
    parser.add_argument('--save_every', type=int, default=1, help='Simpan 1 dari N frame (1 = semua frame)')
    args = parser.parse_args()

    print("Args:", args)
    ensure_dir(args.outdir)
    out_path = os.path.join(args.outdir, f"{args.label}.csv")
    first_time = not os.path.exists(out_path)
    print("Output file:", out_path, "First time:", first_time)

    try:
        print("Loading YOLO model (this may take a while on first run)...")
        yolo = YOLO("yolov8n.pt")
        print("YOLO loaded.")
    except Exception as e:
        print("Failed to load YOLO model:", e)
        traceback.print_exc()
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # === Perbaikan terbesar ada di sini ===
    print(f"\nOpening webcam dengan ID preferensi {args.webcam} ...")
    cap = open_webcam_with_debug(args.webcam)

    if cap is None:
        print("Gagal membuka webcam. Stop.")
        return

    header = build_header()
    f = open(out_path, 'a', newline='')
    writer = csv.writer(f)
    if first_time:
        writer.writerow(header)
        f.flush()

    print(f"Mulai rekam label='{args.label}' selama {args.duration} detik. Tekan 'q' untuk stop lebih awal.")

    start = time.time()
    saved = 0
    frame_count = 0

    try:
        while time.time() - start < args.duration:
            ret, frame = cap.read()
            if not ret:
                print("Frame gagal dibaca (ret=False). Retry...")
                time.sleep(0.05)
                continue

            frame_count += 1

            try:
                results = yolo(frame)
            except Exception as e:
                print("YOLO inference error:", e)
                traceback.print_exc()
                break

            res = results[0]
            boxes = []
            if hasattr(res, 'boxes') and len(res.boxes) > 0:
                for box in res.boxes:
                    try:
                        cls = int(box.cls[0])
                        if cls == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            boxes.append((x1, y1, x2, y2))
                    except:
                        continue

            disp = frame.copy()
            for (x1, y1, x2, y2) in boxes:
                x1c, y1c = max(x1, 0), max(y1, 0)
                x2c, y2c = min(x2, frame.shape[1]-1), min(y2, frame.shape[0]-1)
                if x2c <= x1c or y2c <= y1c:
                    continue

                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                res_pose = pose.process(rgb)
                if not res_pose.pose_landmarks:
                    cv2.rectangle(disp, (x1c, y1c), (x2c, y2c), (0,128,255), 1)
                    continue

                lm = res_pose.pose_landmarks.landmark

                if (frame_count % max(1, args.save_every)) == 0:
                    row = []
                    for i in range(NUM_LANDMARKS):
                        row += [lm[i].x, lm[i].y, lm[i].z, lm[i].visibility]
                    row += [args.label, time.time()]
                    writer.writerow(row)
                    saved += 1

                    if saved % 10 == 0:
                        f.flush()

                cv2.rectangle(disp, (x1c, y1c), (x2c, y2c), (0,255,0), 2)
                cv2.putText(disp, f"Saved: {saved}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            cv2.imshow("Collector (press q to stop)", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stop requested by user (q).")
                break

    finally:
        print("Cleaning up...")
        f.flush()
        f.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Total sampel disimpan: {saved}. File: {out_path}")


if __name__ == "__main__":
    main()

print("Script is running")

from ultralytics import YOLO
import cv2

def main(webcam_id=0, conf=0.4):
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(webcam_id, cv2.CAP_DSHOW)
    print("Cap opened?", cap.isOpened())
    if not cap.isOpened():
        print("Webcam tidak terdeteksi.")
        return

    print("Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        print("Frame?", ret)
        if not ret:
            break

        results = model(frame)

        cv2.imshow("TEST WINDOW", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2

for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_V4L2]:
    cap = cv2.VideoCapture(0, backend)
    print(backend, "opened:", cap.isOpened())
    cap.release()

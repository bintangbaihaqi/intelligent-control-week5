from ultralytics import YOLO
import mediapipe as mp
import cv2
import numpy as np

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Buka video menggunakan OpenCV
cap = cv2.VideoCapture("video orang-orang dewasa.mp4")

# Dapatkan informasi video (lebar, tinggi, FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Inisialisasi VideoWriter untuk menyimpan video
out = cv2.VideoWriter(
    "output_video.mp4",  # Nama file output
    cv2.VideoWriter_fourcc(*"mp4v"),  # Codec video
    fps,  # Frame rate
    (frame_width, frame_height)  # Resolusi video
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke RGB untuk MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 🔍 Deteksi tangan dengan MediaPipe (maksimal 2 tangan)
    results_hands = hands.process(frame_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Deteksi pose menggunakan YOLO pada frame
    results = model(frame)

    # Tambahkan anotasi deteksi pose pada frame
    for result in results:
        frame = result.plot()

    # Tampilkan frame dengan anotasi
    cv2.imshow("YOLO + MediaPipe Hands", frame)

    # Simpan frame ke video output
    out.write(frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan resource
cap.release()
out.release()
cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Konversi frame ke RGB untuk MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mode Night Vision: Konversi frame ke skala abu-abu dan tambahkan colormap
    night_vision = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(night_vision, cv2.COLORMAP_JET)

    # Deteksi pose menggunakan YOLO pada frame asli
    results_frame = model(frame)

    # Deteksi pose menggunakan YOLO pada frame night vision
    results_night = model(night_vision)

    # 🔍 Deteksi tangan dengan MediaPipe (maksimal 2 tangan)
    results_hands = hands.process(frame_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 🔍 Deteksi tangan dengan MediaPipe (maksimal 2 tangan)
    results_hands = hands.process(frame_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(night_vision, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tambahkan anotasi deteksi pose pada frame asli
    for result in results_frame:
        frame = result.plot()  # Tambahkan anotasi pada frame asli

    # Tambahkan anotasi deteksi pose pada frame night vision
    for result in results_night:
        night_vision = result.plot()  # Tambahkan anotasi pada frame night vision

    # Gabungkan frame asli dengan night vision secara horizontal
    combined_frame = np.hstack((frame, night_vision))

    # Tampilkan frame gabungan
    cv2.imshow("Combined Frame (YOLO + Night Vision + MediaPipe Hands)", combined_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
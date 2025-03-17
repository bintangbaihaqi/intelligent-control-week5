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

# Baca gambar
frame = cv2.imread("foto orang dewasa.jpeg")

# Konversi frame ke RGB untuk MediaPipe
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 🔍 Deteksi tangan dengan MediaPipe (maksimal 2 tangan)
results_hands = hands.process(frame_rgb)
if results_hands.multi_hand_landmarks:
    for hand_landmarks in results_hands.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Deteksi pose pada gambar menggunakan YOLO
results = model(frame, show=True)

# Simpan hasil
results[0].save("foto orang dewasa_output.jpeg")
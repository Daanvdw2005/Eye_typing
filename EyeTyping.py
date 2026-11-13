import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# 1. Model automatisch downloaden als het nog niet in je map staat
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
  print("Bezig met downloaden van het MediaPipe Face Landmarker model...")
  url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
  urllib.request.urlretrieve(url, MODEL_PATH)
  print("Download voltooid!")

# 2. Nieuwe MediaPipe Tasks API (FaceLandmarker) initialiseren
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(options)

# Toetsenbord en status variabelen
keys = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ_<")
current_key_index = 0
typed_text = ""
last_scan_time = time.time()
scan_speed = 0.8  # Seconden per letter

# Variabelen voor bewust knipperen
blink_start = None
BLINK_HOLD_TIME = 0.30  # Ogen 0.3s sluiten = typen


def get_eye_ratio(landmarks, top_idx, bottom_idx, left_idx, right_idx, w, h):
  top = np.array([landmarks[top_idx].x * w, landmarks[top_idx].y * h])
  bottom = np.array([landmarks[bottom_idx].x * w, landmarks[bottom_idx].y * h])
  left = np.array([landmarks[left_idx].x * w, landmarks[left_idx].y * h])
  right = np.array([landmarks[right_idx].x * w, landmarks[right_idx].y * h])

  vertical_dist = np.linalg.norm(top - bottom)
  horizontal_dist = np.linalg.norm(left - right)
  return vertical_dist / horizontal_dist


cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Converteer naar MediaPipe Image en voer detectie uit met timestamp
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
  timestamp_ms = int((time.time() - start_time) * 1000)
  results = detector.detect_for_video(mp_image, timestamp_ms)

  # Automatisch scannen door het toetsenbord
  if time.time() - last_scan_time > scan_speed:
    current_key_index = (current_key_index + 1) % len(keys)
    last_scan_time = time.time()

  # Oog-analyse
  if results.face_landmarks:
    landmarks = results.face_landmarks[0]

    # Bereken hoe ver het oog open is (159=boven, 145=onder, 33=links, 133=rechts)
    ear = get_eye_ratio(landmarks, 159, 145, 33, 133, w, h)

    # Toon de actuele EAR-waarde rechtsboven zodat je makkelijk kunt kalibreren
    cv2.putText(
        frame,
        f"Oog-ratio (EAR): {ear:.2f}",
        (w - 280, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # Drempelwaarde voor gesloten oog (pas 0.20 aan als jouw waarde hoger/lager ligt)
    if ear < 0.20:
      if blink_start is None:
        blink_start = time.time()
      elif time.time() - blink_start >= BLINK_HOLD_TIME:
        selected = keys[current_key_index]
        if selected == "_":
          typed_text += " "
        elif selected == "<":
          typed_text = typed_text[:-1]
        else:
          typed_text += selected

        blink_start = time.time() + 0.8  # Korte pauze na het typen van 1 letter
    else:
      blink_start = None

  # Visuele interface tekenen op het scherm
  cv2.rectangle(frame, (20, 20), (w - 300, 90), (30, 30, 30), -1)
  cv2.putText(
      frame,
      f"Tekst: {typed_text}",
      (35, 65),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.9,
      (255, 255, 255),
      2,
  )

  # Teken de knoppen onderaan
  cols = 14
  key_w = (w - 40) // cols
  for i, key in enumerate(keys):
    r = i // cols
    c = i % cols
    x = 20 + c * key_w
    y = h - 130 + r * 55

    bg_color = (0, 200, 0) if i == current_key_index else (60, 60, 60)
    cv2.rectangle(frame, (x, y), (x + key_w - 5, y + 45), bg_color, -1)
    cv2.putText(
        frame,
        key,
        (x + 10, y + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

  cv2.imshow("Eye Typer - Sluit ogen (0.3s) om te typen | Q = Stop", frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

detector.close()
cap.release()
cv2.destroyAllWindows()
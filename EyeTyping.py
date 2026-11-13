import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
  url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
  urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(options)

keys = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ_<")
current_key_index = 13  # Start in het midden van het alfabet
typed_text = ""

# Timing voor oogbeweging en knipperen
last_move_time = time.time()
MOVE_COOLDOWN = 0.35  # Hoe snel je door de letters wandelt als je opzij kijkt
blink_start = None
BLINK_HOLD_TIME = 0.28


def get_eye_ratio(landmarks, top_idx, bottom_idx, left_idx, right_idx, w, h):
  top = np.array([landmarks[top_idx].x * w, landmarks[top_idx].y * h])
  bottom = np.array([landmarks[bottom_idx].x * w, landmarks[bottom_idx].y * h])
  left = np.array([landmarks[left_idx].x * w, landmarks[left_idx].y * h])
  right = np.array([landmarks[right_idx].x * w, landmarks[right_idx].y * h])
  return np.linalg.norm(top - bottom) / np.linalg.norm(left - right)


def get_gaze_ratio(landmarks, w, h):
  """Berekent de horizontale positie van de iris tussen de ooghoeken (0.0 = links, 0.5 = midden, 1.0 = rechts)."""
  # Linkeroog hoeken (33, 133) en iris-centrum (468)
  l_corner1 = landmarks[33].x * w
  l_corner2 = landmarks[133].x * w
  l_iris = landmarks[468].x * w
  ratio_left = (l_iris - min(l_corner1, l_corner2)) / (
      abs(l_corner2 - l_corner1) + 1e-6
  )

  # Rechteroog hoeken (362, 263) en iris-centrum (473)
  r_corner1 = landmarks[362].x * w
  r_corner2 = landmarks[263].x * w
  r_iris = landmarks[473].x * w
  ratio_right = (r_iris - min(r_corner1, r_corner2)) / (
      abs(r_corner2 - r_corner1) + 1e-6
  )

  return (ratio_left + ratio_right) / 2.0


cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  frame = cv2.flip(frame, 1)
  h, w, _ = frame.shape
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
  timestamp_ms = int((time.time() - start_time) * 1000)
  results = detector.detect_for_video(mp_image, timestamp_ms)

  gaze_status = "MIDDEN"
  gaze_color = (200, 200, 200)

  if results.face_landmarks:
    landmarks = results.face_landmarks[0]

    # Teken groene cirkels op beide pupillen (468 en 473) zodat je de tracking ziet
    for iris_idx in [468, 473]:
      ix, iy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
      cv2.circle(frame, (ix, iy), 4, (0, 255, 0), -1)

    ear = get_eye_ratio(landmarks, 159, 145, 33, 133, w, h)
    gaze = get_gaze_ratio(landmarks, w, h)

    # Debug info rechtsboven
    cv2.putText(
        frame,
        f"EAR: {ear:.2f} | Gaze: {gaze:.2f}",
        (w - 310, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )

    # 1. Check eerst of de ogen gesloten zijn (Knipperen = Typen)
    if ear < 0.20:
      gaze_status = "KNIPPER..."
      gaze_color = (0, 165, 255)
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
        blink_start = time.time() + 0.7  # Voorkom dubbel typen
    else:
      blink_start = None

      # 2. Alleen sturen met je ogen als je ogen OPEN zijn
      # Pas 0.44 (links) en 0.58 (rechts) aan op basis van jouw 'Gaze'-getal rechtsboven!
      if gaze < 0.44:
        gaze_status = "<< KIJKT LINKS"
        gaze_color = (255, 200, 0)
        if time.time() - last_move_time > MOVE_COOLDOWN:
          current_key_index = (current_key_index - 1) % len(keys)
          last_move_time = time.time()
      elif gaze > 0.56:
        gaze_status = "KIJKT RECHTS >>"
        gaze_color = (255, 200, 0)
        if time.time() - last_move_time > MOVE_COOLDOWN:
          current_key_index = (current_key_index + 1) % len(keys)
          last_move_time = time.time()

  # Tekstvak en kijk-richting tonen
  cv2.rectangle(frame, (20, 15), (w - 330, 85), (30, 30, 30), -1)
  cv2.putText(
      frame,
      f"Tekst: {typed_text}",
      (35, 60),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.85,
      (255, 255, 255),
      2,
  )
  cv2.putText(
      frame,
      gaze_status,
      (w - 310, 80),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      gaze_color,
      2,
  )

  # Toetsenbord tekenen
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

  cv2.imshow(
      "Eye Typer - Kijk Links/Rechts om te sturen | Knipper om te typen", frame
  )
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

detector.close()
cap.release()
cv2.destroyAllWindows()
import cv2
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from threading import Thread
from insightface.app import FaceAnalysis
from utils.anti_spoof_predictor import AntiSpoofPredictor
import os

# Constants
THRESHOLD = 0.6
LIVENESS_THRESHOLD = 0.35
ATTENDANCE_CSV = "attendance_log.csv"
ATTENDANCE_TIME_LIMIT = datetime.strptime("09:30:00", "%H:%M:%S").time()

# Load embeddings
with open("embeddings/employee_embeddings.pkl", "rb") as f:
    db = pickle.load(f)

# Load previous attendance
if os.path.exists(ATTENDANCE_CSV):
    attendance_log = pd.read_csv(ATTENDANCE_CSV).to_dict(orient="records")
else:
    attendance_log = []

# InsightFace & Spoof Detector
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
spoof_detector = AntiSpoofPredictor("anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth")

# Cosine Similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Face Recognition
def recognize_face(embedding):
    best_match, best_score = None, 0
    for name, db_embed in db.items():
        sim = cosine_similarity(embedding, db_embed)
        if sim > THRESHOLD and sim > best_score:
            best_score = sim
            best_match = name
    return best_match

# Fancy Box Drawing
def draw_fancy_face_box(frame, bbox, color=(57, 255, 20), thickness=2):
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
    cv2.ellipse(frame, center, axes, 0, 0, 360, color, thickness)

# Top-Center Message
def draw_top_message(frame, text, color=(57, 255, 20), bg_color=(0, 0, 0), alpha=0.6):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    box_w, box_h = 460, 50
    top_left = (w // 2 - box_w // 2, 30)
    bottom_right = (top_left[0] + box_w, top_left[1] + box_h)
    cv2.rectangle(overlay, top_left, bottom_right, bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (top_left[0] + 20, top_left[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

# Process each frame
def process_frame(frame, faces, today_logs):
    global attendance_log
    message = ""

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        live_score = spoof_detector.predict(face_crop)
        if isinstance(live_score, np.ndarray):
            live_score = float(live_score)

        if live_score < LIVENESS_THRESHOLD:
            draw_fancy_face_box(frame, (x1, y1, x2, y2), color=(0, 0, 255))
            message = "Spoof Detected"
            continue

        emb = face.embedding
        name = recognize_face(emb)

        if name:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            current_time = now.time()

            marked_today = any(log["name"] == name and log["timestamp"].startswith(today) for log in attendance_log)

            if marked_today:
                message = f"{name}: Already Marked"
            else:
                status = "Late" if current_time > ATTENDANCE_TIME_LIMIT else "On Time"
                attendance_log.append({"name": name, "timestamp": timestamp, "status": status})
                message = f"{name}: {status} Marked"

            draw_fancy_face_box(frame, (x1, y1, x2, y2))
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (57, 255, 20), 2)
        else:
            draw_fancy_face_box(frame, (x1, y1, x2, y2), color=(0, 140, 255))
            message = "Unknown Face"

    if message:
        draw_top_message(frame, message)

# Main loop
cap = cv2.VideoCapture(0)
print("Starting recognition... Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        faces = app.get(frame)
        today_logs = [log for log in attendance_log if log["timestamp"].startswith(datetime.now().strftime("%Y-%m-%d"))]

        thread = Thread(target=process_frame, args=(frame, faces, today_logs))
        thread.start()
        thread.join()

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    pd.DataFrame(attendance_log).to_csv(ATTENDANCE_CSV, index=False)
    print("[INFO] Attendance saved.")

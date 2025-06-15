# recognize_and_log.py
import cv2
import time
import pickle
import numpy as np
from datetime import datetime
import pandas as pd
from insightface.app import FaceAnalysis
from utils.anti_spoof_predictor import AntiSpoofPredictor  # 👈 Import spoof module

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load embeddings
with open("embeddings/employee_embeddings.pkl", "rb") as f:
    db = pickle.load(f)

# Initialize anti-spoof model
spoof_detector = AntiSpoofPredictor("anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth")

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Face recognition function
def recognize_face(embedding, threshold=0.6):
    best_match = None
    best_score = 0
    for name, db_embed in db.items():
        sim = cosine_similarity(embedding, db_embed)
        if sim > threshold and sim > best_score:
            best_score = sim
            best_match = name
    return best_match

# Start webcam
cap = cv2.VideoCapture(0)
attendance_log = []

print("Starting recognition... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Spoof Detection 🔒
    live_score = spoof_detector.predict(frame)
    if isinstance(live_score, np.ndarray):
        live_score = float(live_score)
    if live_score < 0.3:
        print("[WARNING] Spoof detected. Skipping recognition.")
        cv2.putText(frame, "Spoof Detected!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Proceed with recognition
    faces = app.get(frame)
    for face in faces:
        emb = face.embedding
        name = recognize_face(emb)
        box = face.bbox.astype(int)
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)

        if name:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log = {"name": name, "timestamp": timestamp}
            if log not in attendance_log:
                attendance_log.append(log)
                print(f"[LOGGED] {log}")
            cv2.putText(frame, name, tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print(f"[DEBUG] Live score: {live_score:.4f}")

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV (replace with Supabase if needed)
df = pd.DataFrame(attendance_log)
df.to_csv("attendance_log.csv", index=False)

# recognize_and_log.py
import cv2
import time
import pickle
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

with open("embeddings/employee_embeddings.pkl", "rb") as f:
    db = pickle.load(f)

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(embedding, threshold=0.6):
    best_match = None
    best_score = 0
    for name, db_embed in db.items():
        sim = cosine_similarity(embedding, db_embed)
        if sim > threshold and sim > best_score:
            best_score = sim
            best_match = name
    return best_match

cap = cv2.VideoCapture(0)
attendance_log = []

print("Starting recognition... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
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

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV (can replace with Supabase API call)
import pandas as pd
df = pd.DataFrame(attendance_log)
df.to_csv("attendance_log.csv", index=False)

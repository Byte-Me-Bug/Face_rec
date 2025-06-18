import cv2
import os
import pickle
import numpy as np
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from utils.anti_spoof_predictor import AntiSpoofPredictor

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

embedder = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
embedder.prepare(ctx_id=0, det_size=(640, 640))

spoof_detector = AntiSpoofPredictor("anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth")

def register_face(emp_name):
    cap = cv2.VideoCapture(0)
    print("[INFO] Press SPACE to capture face, or Q to quit...")

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Webcam not accessible.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x1 = int(bboxC.xmin * iw)
                    y1 = int(bboxC.ymin * ih)
                    x2 = x1 + int(bboxC.width * iw)
                    y2 = y1 + int(bboxC.height * ih)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Capture Face", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    if not results.detections:
        print("[ERROR] No face detected.")
        return

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        print("[ERROR] Cropped face is empty.")
        return

    live_score = spoof_detector.predict(face_crop)
    print(f"[INFO] Live confidence score: {float(live_score):.4f}")
    if live_score < 0.80:
        print("[WARNING] Spoof detected! Registration aborted.")
        return

    faces = embedder.get(frame)
    if not faces:
        print("[ERROR] InsightFace couldn't extract embedding.")
        return

    embedding = faces[0].normed_embedding

    os.makedirs("embeddings", exist_ok=True)
    db_path = "embeddings/employee_embeddings.pkl"
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            db = pickle.load(f)
    else:
        db = {}

    for existing_name, existing_emb in db.items():
        similarity = cosine_similarity([embedding], [existing_emb])[0][0]
        if similarity > 0.60:
            print(f"[WARNING] Face already registered as '{existing_name}' (similarity: {similarity:.2f})")
            return

    db[emp_name] = embedding
    with open(db_path, "wb") as f:
        pickle.dump(db, f)

    os.makedirs("known_faces", exist_ok=True)
    img_path = os.path.join("known_faces", emp_name + ".jpg")
    cv2.imwrite(img_path, frame)
    print(f"[SUCCESS] Face of '{emp_name}' registered successfully.")

# Entry point
if __name__ == "__main__":
    emp_name = input("Enter employee name (no spaces): ").strip()
    if " " in emp_name or not emp_name:
        print("[ERROR] Name must be non-empty and without spaces.")
    else:
        register_face(emp_name)

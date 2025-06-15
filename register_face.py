import cv2
import insightface
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from utils.anti_spoof_predictor import AntiSpoofPredictor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Initialize spoof detection model
spoof_detector = AntiSpoofPredictor("anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth")

def register_face(emp_name):
    # Initialize face analysis model
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Open webcam
    cap = cv2.VideoCapture(0)
    print("[INFO] Press SPACE to capture face, or Q to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not access webcam.")
            break

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

    # Detect face
    faces = app.get(frame)
    if not faces:
        print("[ERROR] No face detected.")
        return

    face = faces[0]

    # Crop face from frame for spoof detection
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    face_crop = frame[y1:y2, x1:x2]

    # Validate cropped region size
    if face_crop.size == 0:
        print("[ERROR] Cropped face is empty.")
        return

    # Run anti-spoof prediction on cropped face
    live_score = spoof_detector.predict(face_crop)
    print(f"[INFO] Live confidence score:  {float(live_score):.4f}")

    if live_score < 0.80:
        print("[WARNING] Spoof detected! Registration aborted.")
        return

    # Proceed to register embedding
    new_embedding = face.normed_embedding

    # Load existing embeddings
    os.makedirs("embeddings", exist_ok=True)
    db_path = "embeddings/employee_embeddings.pkl"
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            db = pickle.load(f)
    else:
        db = {}

    # Check for duplicates
    for existing_name, existing_emb in db.items():
        similarity = cosine_similarity([new_embedding], [existing_emb])[0][0]
        if similarity > 0.60:
            print(f"[WARNING] Face already registered as '{existing_name}' (similarity: {similarity:.2f})")
            return

    # Save embedding
    db[emp_name] = new_embedding
    with open(db_path, "wb") as f:
        pickle.dump(db, f)

    # Save captured image
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

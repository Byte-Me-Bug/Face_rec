# register_face.py
import os
import cv2
import pickle
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def capture_face_from_webcam():
    cap = cv2.VideoCapture(0)
    print("[INFO] Press SPACE to capture, or Q to quit...")

    while True:
        ret, frame = cap.read()
        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1)

        if key == ord(' '):  # Space to capture
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

def register_face(name):
    frame = capture_face_from_webcam()
    if frame is None:
        print("[INFO] Capture cancelled.")
        return

    faces = app.get(frame)
    if not faces:
        print("[ERROR] No face found in captured image!")
        return

    embedding = faces[0].embedding

    # Load existing embeddings
    os.makedirs("embeddings", exist_ok=True)
    if os.path.exists("embeddings/employee_embeddings.pkl"):
        with open("embeddings/employee_embeddings.pkl", "rb") as f:
            db = pickle.load(f)
    else:
        db = {}

    db[name] = embedding
    with open("embeddings/employee_embeddings.pkl", "wb") as f:
        pickle.dump(db, f)

    # Optionally save the captured image
    os.makedirs("known_faces", exist_ok=True)
    cv2.imwrite(f"known_faces/{name}.jpg", frame)
    print(f"[SUCCESS] {name} registered and image saved!")

# Example: manually enter name
if __name__ == "__main__":
    emp_name = input("Enter employee name (no spaces): ")
    register_face(emp_name)

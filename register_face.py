import cv2
import numpy as np
from utils.anti_spoof_predictor import AntiSpoofPredictor
import os
import warnings
import logging

# Suppress logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('absl').setLevel(logging.ERROR)

# Load the improved anti-spoof model
spoof_detector = AntiSpoofPredictor("anti_spoof_models/4_2_0_128x128_MiniFASNetV2.pth")

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press SPACE to capture and check for spoof...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not accessible.")
        break

    # Show the live feed
    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        # Crop the center of the image (focus on face-like region)
        h, w = frame.shape[:2]
        size = min(h, w)
        x1 = w // 2 - size // 2
        y1 = h // 2 - size // 2
        cropped = frame[y1:y1+size, x1:x1+size]
        resized = cv2.resize(cropped, (128, 128))

        # Check spoof score
        live_score = spoof_detector.predict(resized)
        print(f"[INFO] Live confidence score: {live_score:.4f}")
        if live_score >= 0.80:
            print("[SUCCESS] Real face detected.")
        else:
            print("[WARNING] Spoof detected!")
        break

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

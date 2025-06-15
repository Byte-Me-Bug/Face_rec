import torch
import torch.nn.functional as F
import cv2
import numpy as np
from utils.models.mfasnet import MiniFASNetV1SE


class AntiSpoofPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MiniFASNetV1SE(input_size=(80, 80))
        state_dict = torch.load(model_path, map_location=self.device)

        # Remove "module." if present in keys
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device).eval()

    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (80, 80))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        input_tensor = torch.from_numpy(img).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()
            return probabilities[0][1]  # Probability of being real

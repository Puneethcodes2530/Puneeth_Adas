from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import cv2

class CLIPSceneDetector:

    def __init__(self):
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("models/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("models/clip-vit-base-patch32")

        # ✅ Define scene prompts
        self.labels = [
            "clear road",
            "night driving",
            "foggy road",
            "rainy road",
            "bright glare sunlight",
            "dusty environment"
        ]

        print("CLIP ready ✅")

    def analyze(self, frame):
        # Convert OpenCV → PIL
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Prepare inputs
        inputs = self.processor(
            text=self.labels,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Run model
        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)[0]

        # Get best label
        idx = probs.argmax().item()
        condition = self._map_label(self.labels[idx])
        confidence = probs[idx].item()

        return condition, confidence

    def _map_label(self, text):
        mapping = {
            "clear road": "CLEAR",
            "night driving": "NIGHT",
            "foggy road": "FOG",
            "rainy road": "RAIN",
            "bright glare sunlight": "GLARE",
            "dusty environment": "DUST"
        }
        return mapping.get(text, "CLEAR")

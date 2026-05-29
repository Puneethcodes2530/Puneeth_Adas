"""
NeuroSentinel v3 — CLIP Scene Analyzer

Scene/weather classification using CLIP.

Features:
✓ Uses local CLIP model
✓ Supports CPU and CUDA automatically
✓ Caches result to avoid running CLIP every frame
✓ Runs CLIP only every N frames
✓ Compatible with existing analyze(frame) calls
✓ Returns: condition, confidence
"""

import os
import time
import cv2
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPSceneDetector:
    """
    CLIP-based scene/weather detector.

    Supported output conditions:
    CLEAR
    NIGHT
    FOG
    RAIN
    GLARE
    DUST
    """

    def __init__(
        self,
        model_dir="models/clip-vit-base-patch32",
        update_every=15,
        device=None
    ):

        print("[INFO] Loading CLIP scene detector...")

        # ----------------------------------------------------
        # Device selection
        # ----------------------------------------------------
        if device is None:

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        else:

            self.device = torch.device(device)

        # ----------------------------------------------------
        # Model path check
        # ----------------------------------------------------
        if not os.path.exists(model_dir):

            raise FileNotFoundError(
                f"CLIP model folder not found: {model_dir}"
            )

        # ----------------------------------------------------
        # Load processor and model
        # ----------------------------------------------------
        self.processor = CLIPProcessor.from_pretrained(
            model_dir
        )

        self.model = CLIPModel.from_pretrained(
            model_dir
        )

        self.model.to(self.device)
        self.model.eval()

        # ----------------------------------------------------
        # Prompt labels
        # ----------------------------------------------------
        self.labels = [
            "clear road",
            "night driving",
            "foggy road",
            "rainy road",
            "bright glare sunlight",
            "dusty environment"
        ]

        self.label_to_condition = {
            "clear road": "CLEAR",
            "night driving": "NIGHT",
            "foggy road": "FOG",
            "rainy road": "RAIN",
            "bright glare sunlight": "GLARE",
            "dusty environment": "DUST"
        }

        # ----------------------------------------------------
        # Cache control
        # ----------------------------------------------------
        self.update_every = max(1, int(update_every))
        self.frame_count = 0

        self.cached_condition = "CLEAR"
        self.cached_confidence = 1.0

        self.last_processing_ms = 0.0

        print(
            f"[INFO] CLIP ready on {self.device}. "
            f"Running every {self.update_every} frames."
        )

    # ========================================================
    # MOVE INPUTS TO DEVICE
    # ========================================================

    def _to_device(self, inputs):

        return {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

    # ========================================================
    # MAP PROMPT TO CONDITION
    # ========================================================

    def _map_label(self, text):

        return self.label_to_condition.get(
            text,
            "CLEAR"
        )

    # ========================================================
    # FULL CLIP INFERENCE
    # ========================================================

    def _run_clip(self, frame):

        start = time.perf_counter()

        # ----------------------------------------------------
        # OpenCV BGR → RGB → PIL
        # ----------------------------------------------------
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        # ----------------------------------------------------
        # Prepare CLIP inputs
        # ----------------------------------------------------
        inputs = self.processor(
            text=self.labels,
            images=image,
            return_tensors="pt",
            padding=True
        )

        inputs = self._to_device(inputs)

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------
        with torch.inference_mode():

            outputs = self.model(**inputs)

            probs = outputs.logits_per_image.softmax(
                dim=1
            )[0]

        idx = int(
            probs.argmax().item()
        )

        best_label = self.labels[idx]

        condition = self._map_label(
            best_label
        )

        confidence = float(
            probs[idx].item()
        )

        self.last_processing_ms = (
            time.perf_counter() - start
        ) * 1000

        return condition, confidence

    # ========================================================
    # PUBLIC ANALYZE FUNCTION
    # ========================================================

    def analyze(
        self,
        frame,
        force=False
    ):

        """
        Returns:
            condition: str
            confidence: float

        Example:
            condition, conf = clip_detector.analyze(frame)
        """

        if frame is None:

            return self.cached_condition, self.cached_confidence

        self.frame_count += 1

        # ----------------------------------------------------
        # Use cached result for most frames
        # ----------------------------------------------------
        should_update = (
            force or
            self.frame_count == 1 or
            self.frame_count % self.update_every == 0
        )

        if not should_update:

            return (
                self.cached_condition,
                self.cached_confidence
            )

        try:

            condition, confidence = self._run_clip(
                frame
            )

            self.cached_condition = condition
            self.cached_confidence = confidence

            return condition, confidence

        except Exception as e:

            print(
                f"[WARNING] CLIP scene detection failed: {e}"
            )

            return (
                self.cached_condition,
                self.cached_confidence
            )

    # ========================================================
    # OPTIONAL SUMMARY
    # ========================================================

    def summary(self):

        return (
            f"CLIPSceneDetector("
            f"condition={self.cached_condition}, "
            f"confidence={self.cached_confidence:.2f}, "
            f"time={self.last_processing_ms:.1f}ms, "
            f"device={self.device})"
        )


# ============================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================
#
# If older files import:
# from perception.scene_analyzer import SceneDetector
#
# this will still work.
# ============================================================

class SceneDetector(CLIPSceneDetector):
    pass
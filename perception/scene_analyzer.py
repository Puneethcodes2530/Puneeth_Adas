"""
NeuroSentinel v3 — CLIP Scene Analyzer

Scene/weather classification using CLIP.

Features:
✓ Uses local CLIP model
✓ Supports CPU and CUDA automatically
✓ Uses FP16 on CUDA to reduce VRAM usage
✓ Caches result to avoid running CLIP every frame
✓ Runs CLIP only every N frames
✓ Compatible with existing AdaptiveDetector scene pipeline
✓ Returns SceneState object from analyze(frame)
"""

import os
import time
import cv2
import torch

from PIL import Image
from dataclasses import dataclass
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# SCENE STATE
# ============================================================

@dataclass
class SceneState:
    """
    Structured scene output used by adaptive detection pipeline.
    """

    condition: str
    severity: float
    confidence: float
    processing_ms: float

    @property
    def conf_threshold(self) -> float:
        """
        Dynamic YOLO confidence threshold based on scene condition.
        Worse visibility → higher threshold.
        """

        return {
            "CLEAR": 0.35,
            "NIGHT": 0.45,
            "FOG": 0.50,
            "RAIN": 0.45,
            "GLARE": 0.55,
            "DUST": 0.50,
        }.get(
            self.condition,
            0.35
        )

    @property
    def confidence_penalty(self) -> float:
        """
        Penalizes YOLO confidence under degraded conditions.
        """

        return {
            "CLEAR": 1.00,
            "NIGHT": 0.75,
            "FOG": 0.70,
            "RAIN": 0.80,
            "GLARE": 0.50,
            "DUST": 0.65,
        }.get(
            self.condition,
            1.00
        )

    @property
    def depth_weight(self) -> float:
        """
        Controls how much depth should be trusted under scene condition.
        """

        return {
            "CLEAR": 1.00,
            "NIGHT": 0.80,
            "FOG": 0.40,
            "RAIN": 0.60,
            "GLARE": 0.50,
            "DUST": 0.55,
        }.get(
            self.condition,
            1.00
        )

    @property
    def tau_margin_weight(self) -> float:
        """
        Controls how much bbox-growth TTC should be trusted.
        Higher under degraded visibility.
        """

        return {
            "CLEAR": 0.50,
            "NIGHT": 0.70,
            "FOG": 0.90,
            "RAIN": 0.85,
            "GLARE": 0.90,
            "DUST": 0.80,
        }.get(
            self.condition,
            0.50
        )

    def summary(self) -> str:
        return (
            f"[{self.condition}] "
            f"confidence={self.confidence:.2f} "
            f"severity={self.severity:.2f} "
            f"threshold={self.conf_threshold:.2f} "
            f"penalty={self.confidence_penalty:.2f} "
            f"depth_weight={self.depth_weight:.2f} "
            f"time={self.processing_ms:.1f}ms"
        )


# ============================================================
# CLIP SCENE DETECTOR
# ============================================================

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
        model_dir: str = "models/clip-vit-base-patch32",
        update_every: int = 15,
        device=None,
        use_fp16=None
    ):

        print("[INFO] Loading CLIP scene detector...")

        # ----------------------------------------------------
        # DEVICE SELECTION
        # ----------------------------------------------------
        if device is None:

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        else:

            self.device = torch.device(device)

        print("[INFO] CLIP device:", self.device)

        # ----------------------------------------------------
        # FP16 SELECTION
        # ----------------------------------------------------
        if use_fp16 is None:

            self.use_fp16 = (
                self.device.type == "cuda"
            )

        else:

            self.use_fp16 = bool(use_fp16)

        # ----------------------------------------------------
        # MODEL PATH CHECK
        # ----------------------------------------------------
        if not os.path.exists(model_dir):

            raise FileNotFoundError(
                f"CLIP model folder not found: {model_dir}"
            )

        # ----------------------------------------------------
        # LOAD PROCESSOR + MODEL
        # ----------------------------------------------------
        self.processor = CLIPProcessor.from_pretrained(
            model_dir
        )

        self.model = CLIPModel.from_pretrained(
            model_dir
        )

        self.model.to(
            self.device
        )

        if self.use_fp16:

            self.model.half()

        self.model.eval()

        # ----------------------------------------------------
        # PROMPT LABELS
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
        # CACHE CONTROL
        # ----------------------------------------------------
        self.update_every = max(
            1,
            int(update_every)
        )

        self.frame_count = 0

        self.cached_condition = "CLEAR"
        self.cached_confidence = 1.0
        self.last_processing_ms = 0.0

        print(
            f"[INFO] CLIP ready on {self.device} "
            f"(fp16={self.use_fp16}). "
            f"Running every {self.update_every} frames."
        )

    # ========================================================
    # HELPER: BUILD SCENE STATE
    # ========================================================

    def _make_state(
        self,
        condition: str,
        confidence: float,
        processing_ms: float
    ) -> SceneState:

        confidence = float(
            max(
                0.0,
                min(
                    confidence,
                    1.0
                )
            )
        )

        severity = float(
            1.0 - confidence
        )

        return SceneState(
            condition=condition,
            severity=severity,
            confidence=confidence,
            processing_ms=processing_ms
        )

    # ========================================================
    # MOVE INPUTS TO DEVICE
    # ========================================================

    def _to_device(
        self,
        inputs
    ):

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        if (
            self.use_fp16 and
            "pixel_values" in inputs
        ):

            inputs["pixel_values"] = (
                inputs["pixel_values"].half()
            )

        return inputs

    # ========================================================
    # MAP PROMPT TO CONDITION
    # ========================================================

    def _map_label(
        self,
        text: str
    ) -> str:

        return self.label_to_condition.get(
            text,
            "CLEAR"
        )

    # ========================================================
    # FULL CLIP INFERENCE
    # ========================================================

    def _run_clip(
        self,
        frame
    ):

        start = time.perf_counter()

        if frame is None:

            return (
                self.cached_condition,
                self.cached_confidence,
                0.0
            )

        if len(frame.shape) != 3:

            return (
                self.cached_condition,
                self.cached_confidence,
                0.0
            )

        # ----------------------------------------------------
        # OpenCV BGR → RGB → PIL
        # ----------------------------------------------------
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        # ----------------------------------------------------
        # PREPARE CLIP INPUTS
        # ----------------------------------------------------
        inputs = self.processor(
            text=self.labels,
            images=image,
            return_tensors="pt",
            padding=True
        )

        inputs = self._to_device(
            inputs
        )

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------
        with torch.inference_mode():

            outputs = self.model(
                **inputs
            )

            probs = outputs.logits_per_image.softmax(
                dim=1
            )[0]

        idx = int(
            probs.argmax().item()
        )

        best_label = self.labels[
            idx
        ]

        condition = self._map_label(
            best_label
        )

        confidence = float(
            probs[idx].item()
        )

        processing_ms = (
            time.perf_counter() -
            start
        ) * 1000

        self.last_processing_ms = processing_ms

        return (
            condition,
            confidence,
            processing_ms
        )

    # ========================================================
    # PUBLIC ANALYZE FUNCTION
    # ========================================================

    def analyze(
        self,
        frame,
        force: bool = False
    ) -> SceneState:

        """
        Returns:
            SceneState object

        Example:
            scene = clip_detector.analyze(frame)
            print(scene.condition)
            print(scene.conf_threshold)
        """

        if frame is None:

            return self._make_state(
                self.cached_condition,
                self.cached_confidence,
                self.last_processing_ms
            )

        self.frame_count += 1

        should_update = (
            force or
            self.frame_count == 1 or
            self.frame_count % self.update_every == 0
        )

        # ----------------------------------------------------
        # USE CACHED RESULT FOR MOST FRAMES
        # ----------------------------------------------------
        if not should_update:

            return self._make_state(
                self.cached_condition,
                self.cached_confidence,
                0.0
            )

        try:

            condition, confidence, processing_ms = self._run_clip(
                frame
            )

            self.cached_condition = condition
            self.cached_confidence = confidence

            return self._make_state(
                condition,
                confidence,
                processing_ms
            )

        except RuntimeError as e:

            if "out of memory" in str(e).lower():

                print(
                    "[WARNING] CLIP CUDA OOM. "
                    "Falling back to cached scene."
                )

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

            else:

                print(
                    f"[WARNING] CLIP runtime error: {e}"
                )

            return self._make_state(
                self.cached_condition,
                self.cached_confidence,
                self.last_processing_ms
            )

        except Exception as e:

            print(
                f"[WARNING] CLIP scene detection failed: {e}"
            )

            return self._make_state(
                self.cached_condition,
                self.cached_confidence,
                self.last_processing_ms
            )

    # ========================================================
    # OLD-STYLE TUPLE API
    # ========================================================

    def analyze_tuple(
        self,
        frame,
        force: bool = False
    ):

        """
        For old scripts that expect:
            condition, confidence = detector.analyze(frame)

        Use:
            condition, confidence = detector.analyze_tuple(frame)
        """

        scene = self.analyze(
            frame,
            force=force
        )

        return (
            scene.condition,
            scene.confidence
        )

    # ========================================================
    # WARMUP
    # ========================================================

    def warmup(
        self,
        frame=None
    ):

        """
        Optional warmup to initialize CUDA kernels/model path.

        If frame is None, uses a dummy road-like image.
        """

        if frame is None:

            frame = 128 * torch.ones(
                224,
                224,
                3,
                dtype=torch.uint8
            ).numpy()

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

        return self.analyze(
            frame,
            force=True
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self
    ) -> str:

        return (
            f"CLIPSceneDetector("
            f"condition={self.cached_condition}, "
            f"confidence={self.cached_confidence:.2f}, "
            f"time={self.last_processing_ms:.1f}ms, "
            f"device={self.device}, "
            f"fp16={self.use_fp16}, "
            f"update_every={self.update_every})"
        )


# ============================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================
#
# Older files can still import:
# from perception.scene_analyzer import SceneDetector
# ============================================================

class SceneDetector(CLIPSceneDetector):
    pass
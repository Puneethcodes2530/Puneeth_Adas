"""
Adaptive Detection Pipeline

Combines CLIP-based Scene Analyzer + YOLOv8 with dynamic thresholds.

This is the Phase 2 adaptive detection module.

Features:
✓ CLIP scene/weather detection
✓ Scene-aware YOLO confidence thresholding
✓ Confidence penalty under degraded scenes
✓ ADAS class filtering
✓ VRU identification
✓ Geometry-based distance estimate
✓ CUDA-aware YOLO execution
✓ Robust fallback if scene detector fails
"""

import cv2
import numpy as np
import time
import os
import sys
import torch

from dataclasses import dataclass
from typing import List, Optional


# ============================================================
# PROJECT ROOT IMPORT FIX
# ============================================================

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


from perception.scene_analyzer import (
    SceneState,
    CLIPSceneDetector
)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Detection:
    class_name: str
    raw_confidence: float
    adj_confidence: float
    bbox: List[int]
    width_px: int
    height_px: int
    est_distance_m: float
    is_vru: bool


@dataclass
class FrameOutput:
    frame_id: int
    timestamp_ms: float
    scene: SceneState
    detections: List[Detection]
    n_objects: int
    processing_ms: float
    within_budget: bool

    def to_dict(self) -> dict:

        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,

            "scene_condition": self.scene.condition,
            "scene_severity": round(self.scene.severity, 3),
            "scene_confidence": round(self.scene.confidence, 3),
            "conf_threshold": self.scene.conf_threshold,
            "confidence_penalty": self.scene.confidence_penalty,

            "n_objects": self.n_objects,
            "processing_ms": round(self.processing_ms, 1),
            "within_budget": self.within_budget,

            "detections": [
                {
                    "class": d.class_name,
                    "raw_conf": round(d.raw_confidence, 3),
                    "adj_conf": round(d.adj_confidence, 3),
                    "bbox": d.bbox,
                    "width_px": d.width_px,
                    "height_px": d.height_px,
                    "est_distance_m": d.est_distance_m,
                    "is_vru": d.is_vru,
                }
                for d in self.detections
            ]
        }


# ============================================================
# ADAPTIVE DETECTOR
# ============================================================

class AdaptiveDetector:

    VRU_CLASSES = {
        "person",
        "bicycle",
        "motorcycle",
        "dog",
        "cat",
        "horse",
        "cow"
    }

    ADAS_CLASSES = {
        "person",
        "bicycle",
        "motorcycle",
        "car",
        "truck",
        "bus",
        "traffic light",
        "stop sign",
        "dog",
        "cat",
        "horse",
        "cow",
        "rickshaw",
        "autorickshaw"
    }

    REAL_HEIGHTS = {
        "person": 1.75,
        "car": 1.50,
        "truck": 3.50,
        "bus": 3.20,
        "bicycle": 1.10,
        "motorcycle": 1.20,
        "dog": 0.60,
        "cat": 0.30,
        "horse": 1.60,
        "cow": 1.45,
        "rickshaw": 1.60,
        "autorickshaw": 1.60
    }

    COLORS = {
        "person": (255, 50, 50),
        "bicycle": (255, 165, 0),
        "motorcycle": (255, 140, 0),
        "car": (50, 205, 50),
        "truck": (0, 128, 255),
        "bus": (128, 0, 255),
        "traffic light": (255, 255, 0),
        "stop sign": (255, 0, 128),
        "dog": (180, 120, 255),
        "cat": (180, 120, 255),
        "horse": (180, 120, 255),
        "cow": (180, 120, 255),
        "rickshaw": (0, 255, 180),
        "autorickshaw": (0, 255, 180)
    }

    COND_COLORS = {
        "CLEAR": (0, 255, 0),
        "NIGHT": (100, 100, 255),
        "FOG": (200, 200, 200),
        "RAIN": (0, 200, 255),
        "GLARE": (255, 255, 0),
        "DUST": (200, 150, 50),
    }

    def __init__(
        self,
        model_weights: str = "yolov8s.pt",
        budget_ms: float = 40.0,
        focal_length: float = 720.0,
        use_clip: bool = True,
        scene_update_every: int = 15,
        device: Optional[str] = None
    ):

        from ultralytics import YOLO

        print("=" * 60)
        print("Loading AdaptiveDetector")
        print("=" * 60)

        self.model_weights = model_weights
        self.model = YOLO(model_weights)

        # ----------------------------------------------------
        # Device selection
        # ----------------------------------------------------
        if device is None:
            self.device = (
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        print(f"✓ YOLO loaded: {model_weights}")
        print(f"✓ YOLO device: {self.device}")

        self.budget = budget_ms
        self.focal_length = focal_length
        self.frame_id = 0

        self.use_clip = use_clip

        if self.use_clip:

            self.clip = CLIPSceneDetector(
                update_every=scene_update_every
            )

        else:

            self.clip = None

        print("✓ AdaptiveDetector ready")

    # ========================================================
    # FALLBACK CLEAR SCENE
    # ========================================================

    def _default_scene(self) -> SceneState:

        return SceneState(
            condition="CLEAR",
            severity=0.0,
            confidence=1.0,
            processing_ms=0.0
        )

    # ========================================================
    # SCENE OUTPUT NORMALIZER
    # ========================================================

    def _get_scene_state(
        self,
        frame: np.ndarray
    ) -> SceneState:

        if self.clip is None:

            return self._default_scene()

        try:

            scene = self.clip.analyze(frame)

            # New API: analyze() returns SceneState
            if isinstance(scene, SceneState):

                return scene

            # Old API compatibility: analyze() returns tuple
            if isinstance(scene, tuple) and len(scene) == 2:

                condition, confidence = scene

                return SceneState(
                    condition=condition,
                    severity=1.0 - float(confidence),
                    confidence=float(confidence),
                    processing_ms=getattr(
                        self.clip,
                        "last_processing_ms",
                        0.0
                    )
                )

            return self._default_scene()

        except Exception as e:

            print(f"[WARNING] Scene analysis failed: {e}")

            return self._default_scene()

    # ========================================================
    # GEOMETRY DISTANCE
    # ========================================================

    def _estimate_distance(
        self,
        bbox: List[int],
        class_name: str
    ) -> float:

        x1, y1, x2, y2 = bbox

        h_px = max(
            y2 - y1,
            1
        )

        real_h = self.REAL_HEIGHTS.get(
            class_name,
            1.5
        )

        distance = (
            self.focal_length *
            real_h
        ) / h_px

        return round(
            float(
                np.clip(
                    distance,
                    1.0,
                    100.0
                )
            ),
            1
        )

    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(
        self,
        frame: np.ndarray
    ) -> FrameOutput:

        t_total = time.perf_counter()

        self.frame_id += 1

        # ----------------------------------------------------
        # STEP 1: Scene detection
        # ----------------------------------------------------
        scene_state = self._get_scene_state(
            frame
        )

        # ----------------------------------------------------
        # STEP 2: YOLO detection
        # ----------------------------------------------------
        # Run YOLO slightly lower than final threshold so that
        # adjusted-confidence filtering can be applied after
        # scene penalty.
        # ----------------------------------------------------
        model_conf = max(
            0.10,
            scene_state.conf_threshold * 0.70
        )

        results = self.model(
            frame,
            conf=model_conf,
            verbose=False,
            device=self.device
        )

        result = results[0]

        detections = []

        # ----------------------------------------------------
        # STEP 3: Process detections
        # ----------------------------------------------------
        if result.boxes is not None:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                raw_conf = float(
                    box.conf[0].item()
                )

                cls_name = self.model.names[
                    int(box.cls[0].item())
                ]

                if cls_name not in self.ADAS_CLASSES:
                    continue

                adj_conf = (
                    raw_conf *
                    scene_state.confidence_penalty
                )

                # Final adaptive thresholding after penalty
                if adj_conf < scene_state.conf_threshold:
                    continue

                h_px = max(
                    y2 - y1,
                    1
                )

                w_px = max(
                    x2 - x1,
                    1
                )

                dist = self._estimate_distance(
                    [x1, y1, x2, y2],
                    cls_name
                )

                detections.append(
                    Detection(
                        class_name=cls_name,
                        raw_confidence=round(raw_conf, 3),
                        adj_confidence=round(adj_conf, 3),
                        bbox=[x1, y1, x2, y2],
                        width_px=w_px,
                        height_px=h_px,
                        est_distance_m=dist,
                        is_vru=(
                            cls_name in self.VRU_CLASSES
                        )
                    )
                )

        total_ms = (
            time.perf_counter() -
            t_total
        ) * 1000

        return FrameOutput(
            frame_id=self.frame_id,
            timestamp_ms=time.time() * 1000,
            scene=scene_state,
            detections=detections,
            n_objects=len(detections),
            processing_ms=total_ms,
            within_budget=(
                total_ms <= self.budget
            )
        )

    # ========================================================
    # DRAW OUTPUT
    # ========================================================

    def draw(
        self,
        frame: np.ndarray,
        output: FrameOutput
    ) -> np.ndarray:

        vis = frame.copy()

        h, w = vis.shape[:2]

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------
        for det in output.detections:

            x1, y1, x2, y2 = det.bbox

            color = self.COLORS.get(
                det.class_name,
                (200, 200, 200)
            )

            thickness = (
                3
                if det.is_vru
                else 2
            )

            cv2.rectangle(
                vis,
                (x1, y1),
                (x2, y2),
                color,
                thickness
            )

            label = (
                f"{det.class_name} "
                f"{det.adj_confidence:.2f} "
                f"{det.est_distance_m:.1f}m"
            )

            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                1
            )

            y_text = max(
                y1,
                th + 10
            )

            cv2.rectangle(
                vis,
                (x1, y_text - th - 8),
                (x1 + tw + 4, y_text),
                color,
                -1
            )

            cv2.putText(
                vis,
                label,
                (x1 + 2, y_text - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

            if det.is_vru:

                cv2.putText(
                    vis,
                    "VRU",
                    (x1, min(h - 5, y2 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 50, 50),
                    2,
                    cv2.LINE_AA
                )

        # ----------------------------------------------------
        # Scene badge
        # ----------------------------------------------------
        cond = output.scene.condition

        color = self.COND_COLORS.get(
            cond,
            (255, 255, 255)
        )

        badge = (
            f"SCENE: {cond} "
            f"conf={output.scene.confidence:.2f} "
            f"sev={output.scene.severity:.2f}"
        )

        cv2.rectangle(
            vis,
            (8, 8),
            (430, 36),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            vis,
            badge,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Threshold badge
        # ----------------------------------------------------
        thresh_txt = (
            f"THRESH:{output.scene.conf_threshold:.2f} "
            f"PEN:{output.scene.confidence_penalty:.2f}"
        )

        cv2.rectangle(
            vis,
            (8, 40),
            (310, 66),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            vis,
            thresh_txt,
            (12, 59),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Object + latency badge
        # ----------------------------------------------------
        cv2.rectangle(
            vis,
            (w - 250, 8),
            (w - 5, 66),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            vis,
            f"Objects: {output.n_objects}",
            (w - 240, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 200),
            2,
            cv2.LINE_AA
        )

        budget_col = (
            (0, 255, 0)
            if output.within_budget
            else (0, 0, 255)
        )

        cv2.putText(
            vis,
            (
                f"{output.processing_ms:.0f}ms "
                f"{'OK' if output.within_budget else 'SLOW'}"
            ),
            (w - 240, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            budget_col,
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Footer
        # ----------------------------------------------------
        cv2.rectangle(
            vis,
            (0, h - 32),
            (w, h),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            vis,
            (
                f"NeuroSentinel v3 | Frame {output.frame_id} | "
                f"{self.model_weights} | Adaptive Scene-Aware Detection"
            ),
            (10, h - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (170, 170, 170),
            1,
            cv2.LINE_AA
        )

        return vis
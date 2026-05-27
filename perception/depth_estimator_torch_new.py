"""
NeuroSentinel v3 — PyTorch DPT Depth Estimator

This module:
✓ Runs monocular depth estimation using DPT
✓ Uses HuggingFace transformers
✓ Returns structured DepthOutput object
✓ Supports ROI-based sampling
✓ Applies temporal smoothing
✓ Fuses geometry + depth estimation
✓ CPU-compatible
"""

import cv2
import time
import numpy as np

from dataclasses import dataclass

import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation


# ============================================================
# DEPTH OUTPUT
# ============================================================

@dataclass
class DepthOutput:

    depth_map: np.ndarray
    uncertainty: np.ndarray
    processing_ms: float
    model_name: str

    # Temporal smoothing cache
    previous_distance: float = None

    # ========================================================
    # SAMPLE DEPTH AT OBJECT BOUNDING BOX
    # ========================================================

    def sample_at_bbox(self, bbox):

        x1, y1, x2, y2 = bbox

        h, w = self.depth_map.shape

        # ----------------------------------------------------
        # SAFETY CLAMP
        # ----------------------------------------------------
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w - 1))

        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h - 1))

        # ----------------------------------------------------
        # INVALID BOX CHECK
        # ----------------------------------------------------
        if x2 <= x1 or y2 <= y1:

            return {
                'distance_m': 999.0,
                'confidence': 0.0
            }

        # ----------------------------------------------------
        # OBJECT SIZE
        # ----------------------------------------------------
        bbox_h = y2 - y1
        bbox_w = x2 - x1

        # ----------------------------------------------------
        # REAL WORLD HEIGHTS
        # ----------------------------------------------------
        REAL_HEIGHTS = {

            'person': 1.7,
            'car': 1.5,
            'truck': 3.2,
            'bus': 3.0,
            'motorcycle': 1.2,
            'bicycle': 1.1,
            'autorickshaw': 1.6
        }

        class_name = getattr(
            self,
            "current_class",
            "person"
        )

        real_h = REAL_HEIGHTS.get(
            class_name,
            1.7
        )

        # ----------------------------------------------------
        # EFFECTIVE OBJECT SIZE
        # ----------------------------------------------------
        effective_size = (
            0.6 * bbox_h +
            0.4 * bbox_w
        )

        # ----------------------------------------------------
        # PINHOLE CAMERA GEOMETRY
        # ----------------------------------------------------
        FOCAL_LENGTH = 720

        heuristic_dist = (
            FOCAL_LENGTH * real_h
        ) / max(effective_size, 1)

        # ----------------------------------------------------
        # DEPTH ROI
        # ----------------------------------------------------
        foot_y = y2

        y_start = max(
            foot_y - 10,
            0
        )

        roi = self.depth_map[
            y_start:foot_y,
            x1:x2
        ]

        if roi.size == 0:

            return {
                'distance_m': 999.0,
                'confidence': 0.0
            }

        # ----------------------------------------------------
        # ROBUST DEPTH VALUE
        # ----------------------------------------------------
        depth_val = float(
            np.percentile(roi, 20)
        )

        depth_val = max(depth_val, 0.05)

        # ----------------------------------------------------
        # DEPTH-BASED DISTANCE
        # ----------------------------------------------------
        #
        # DPT gives inverse relative depth
        # Higher depth => closer object
        # ----------------------------------------------------
        depth_distance = 45.0 / depth_val

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------
        fused_distance = (
            0.55 * heuristic_dist +
            0.45 * depth_distance
        )

        # ----------------------------------------------------
        # TEMPORAL SMOOTHING
        # ----------------------------------------------------
        if self.previous_distance is not None:

            fused_distance = (
                0.7 * self.previous_distance +
                0.3 * fused_distance
            )

        self.previous_distance = fused_distance

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------
        confidence = min(
            1.0,
            roi.std() / (roi.mean() + 1e-5)
        )

        confidence = 1.0 - confidence
        confidence = max(0.1, confidence)

        return {

            'distance_m': float(fused_distance),

            'confidence': float(confidence)
        }


# ============================================================
# DEPTH ESTIMATOR
# ============================================================

class DepthEstimatorTorch:

    def __init__(self):

        print("[INFO] Loading DPT depth model...")

        self.device = torch.device("cpu")

        self.processor = DPTImageProcessor.from_pretrained(
            "models/dpt_hybrid"
        )

        self.model = DPTForDepthEstimation.from_pretrained(
            "models/dpt_hybrid"
        )

        self.model.to(self.device)

        self.model.eval()

        self.model_name = "DPT-Hybrid-MiDaS"

        print("[INFO] Depth model loaded.")

    # ========================================================
    # ESTIMATE DEPTH
    # ========================================================

    def estimate(self, frame):

        start = time.time()

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        inputs = self.processor(
            images=rgb,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(**inputs)

            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------
        depth_map = cv2.normalize(
            depth_map,
            None,
            0,
            1,
            cv2.NORM_MINMAX
        )

        processing_ms = (
            time.time() - start
        ) * 1000

        return DepthOutput(

            depth_map=depth_map,

            uncertainty=np.zeros_like(depth_map),

            processing_ms=processing_ms,

            model_name=self.model_name
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def visualize(self, frame, depth_output):

        depth_uint8 = (
            depth_output.depth_map * 255
        ).astype(np.uint8)

        colored = cv2.applyColorMap(
            depth_uint8,
            cv2.COLORMAP_MAGMA
        )

        combined = np.hstack([
            frame,
            colored
        ])

        return combined
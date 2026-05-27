# ============================================================
# FILE: perception/depth_estimator_da.py
# ============================================================

"""
NeuroSentinel v3 — Depth Anything v2 Estimator

This module:
✓ Runs monocular depth estimation
✓ Uses Depth Anything v2
✓ Returns structured DepthOutput
✓ Supports ROI sampling
✓ Uses geometry-first distance estimation
✓ Uses depth as refinement only
✓ CPU-compatible
"""

import cv2
import time
import numpy as np

from dataclasses import dataclass

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation
)


# ============================================================
# DEPTH OUTPUT
# ============================================================

@dataclass
class DepthOutput:

    depth_map: np.ndarray
    processing_ms: float
    model_name: str

    # ========================================================
    # SAMPLE DEPTH AT OBJECT BOUNDING BOX
    # ========================================================

    def sample_at_bbox(self, bbox):

        x1, y1, x2, y2 = map(int, bbox)

        h, w = self.depth_map.shape

        # ----------------------------------------------------
        # SAFETY CLAMP
        # ----------------------------------------------------

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))

        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # ----------------------------------------------------
        # INVALID ROI CHECK
        # ----------------------------------------------------

        if x2 <= x1 or y2 <= y1:

            return {
                'distance_m': 999.0,
                'confidence': 0.0
            }

        # ----------------------------------------------------
        # BOTTOM CENTER ROI
        # ----------------------------------------------------
        #
        # Better for road-contact depth estimation
        # ----------------------------------------------------

        cx = (x1 + x2) // 2

        half_w = max(5, (x2 - x1) // 4)

        y_start = int(
            y2 - 0.2 * (y2 - y1)
        )

        roi = self.depth_map[
            y_start:y2,
            max(0, cx - half_w):min(w, cx + half_w)
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
        # REAL-WORLD OBJECT HEIGHTS
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
        # GEOMETRIC DISTANCE
        # ----------------------------------------------------

        bbox_h = y2 - y1
        bbox_w = x2 - x1

        effective_size = (
            0.7 * bbox_h +
            0.3 * bbox_w
        )

        FOCAL_LENGTH = 850

        geometry_distance = (
            FOCAL_LENGTH * real_h
        ) / max(effective_size, 1)

        # ----------------------------------------------------
        # DEPTH REFINEMENT
        # ----------------------------------------------------
        #
        # Depth map only refines geometry.
        # DO NOT let depth dominate.
        # ----------------------------------------------------

        depth_refinement = 1.0 / depth_val

        depth_refinement = np.clip(
            depth_refinement,
            0.7,
            1.3
        )

        # ----------------------------------------------------
        # FINAL DISTANCE
        # ----------------------------------------------------

        distance = (
            geometry_distance *
            depth_refinement
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = 1.0 - min(
            1.0,
            roi.std() / (roi.mean() + 1e-5)
        )

        confidence = max(0.1, confidence)

        return {

            'distance_m': float(distance),

            'confidence': float(confidence)
        }


# ============================================================
# DEPTH ESTIMATOR
# ============================================================

class DepthEstimatorDA:

    def __init__(self):

        print("[INFO] Loading Depth Anything model...")

        self.device = torch.device("cpu")

        self.processor = AutoImageProcessor.from_pretrained(
            "models/depth_anything"
        )

        self.model = AutoModelForDepthEstimation.from_pretrained(
            "models/depth_anything"
        )

        self.model.to(self.device)

        self.model.eval()

        self.model_name = "Depth-Anything-v2"

        print("[INFO] Depth Anything loaded.")

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

            depth = outputs.predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

        depth = depth.cpu().numpy()

        # ----------------------------------------------------
        # NORMALIZE DEPTH
        # ----------------------------------------------------

        depth = (
            depth - depth.min()
        ) / (
            depth.max() - depth.min() + 1e-6
        )

        ms = (
            time.time() - start
        ) * 1000

        return DepthOutput(

            depth_map=depth,

            processing_ms=ms,

            model_name=self.model_name
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def visualize(self, frame, depth_output):

        d = (
            depth_output.depth_map * 255
        ).astype(np.uint8)

        d_col = cv2.applyColorMap(
            d,
            cv2.COLORMAP_MAGMA
        )

        return np.hstack([
            frame,
            d_col
        ])
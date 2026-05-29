# ============================================================
# FILE: perception/depth_estimator.py
# ============================================================

"""
NeuroSentinel v3 — Depth Anything v2 Estimator

This module:
✓ Runs monocular depth estimation using Depth Anything v2
✓ Uses HuggingFace Transformers
✓ Supports CUDA if available
✓ Falls back to CPU safely
✓ Returns structured DepthOutput object
✓ Supports ROI-based object depth sampling
✓ Uses geometry-first distance estimation
✓ Uses depth only as controlled refinement
✓ CPU/GPU compatible
"""

import os
import cv2
import time
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation
)


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

REAL_HEIGHTS = {
    "person": 1.7,
    "car": 1.5,
    "truck": 3.2,
    "bus": 3.0,
    "motorcycle": 1.2,
    "bicycle": 1.1,
    "autorickshaw": 1.6
}

DEFAULT_FOCAL_LENGTH = 850.0


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

    def sample_at_bbox(
        self,
        bbox: List[int],
        class_name: Optional[str] = None,
        focal_length: float = DEFAULT_FOCAL_LENGTH
    ) -> Dict[str, float]:

        """
        Estimate object distance using:
        1. Bounding box geometry
        2. Bottom-center ROI depth refinement
        3. Confidence-gated fusion

        Returns:
            {
                distance_m,
                confidence,
                geometry_distance,
                depth_refinement,
                depth_val
            }
        """

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
        # INVALID BOX CHECK
        # ----------------------------------------------------
        if x2 <= x1 or y2 <= y1:

            return {
                "distance_m": 999.0,
                "confidence": 0.0,
                "geometry_distance": 999.0,
                "depth_refinement": 1.0,
                "depth_val": 0.0
            }

        # ----------------------------------------------------
        # CLASS HANDLING
        # ----------------------------------------------------
        #
        # Supports both:
        # 1. sample_at_bbox(bbox, class_name="car")
        # 2. depth_out.current_class = "car"
        # ----------------------------------------------------
        if class_name is None:

            class_name = getattr(
                self,
                "current_class",
                "car"
            )

        real_h = REAL_HEIGHTS.get(
            class_name,
            1.5
        )

        # ----------------------------------------------------
        # GEOMETRIC DISTANCE
        # ----------------------------------------------------
        bbox_h = max(y2 - y1, 1)
        bbox_w = max(x2 - x1, 1)

        effective_size = (
            0.75 * bbox_h +
            0.25 * bbox_w
        )

        geometry_distance = (
            focal_length * real_h
        ) / max(effective_size, 1)

        geometry_distance = float(
            np.clip(
                geometry_distance,
                1.0,
                100.0
            )
        )

        # ----------------------------------------------------
        # BOTTOM-CENTER ROI
        # ----------------------------------------------------
        #
        # This focuses on object-road contact region.
        # It avoids full bbox averaging, which includes background.
        # ----------------------------------------------------
        cx = (x1 + x2) // 2

        half_w = max(
            4,
            bbox_w // 6
        )

        y_start = int(
            y2 - 0.20 * bbox_h
        )

        x_start = max(
            0,
            cx - half_w
        )

        x_end = min(
            w,
            cx + half_w
        )

        roi = self.depth_map[
            y_start:y2,
            x_start:x_end
        ]

        if roi.size == 0:

            return {
                "distance_m": geometry_distance,
                "confidence": 0.0,
                "geometry_distance": geometry_distance,
                "depth_refinement": 1.0,
                "depth_val": 0.0
            }

        # ----------------------------------------------------
        # VALID DEPTH VALUES
        # ----------------------------------------------------
        valid = roi[
            np.isfinite(roi)
        ]

        valid = valid[
            valid > 0
        ]

        if valid.size == 0:

            return {
                "distance_m": geometry_distance,
                "confidence": 0.0,
                "geometry_distance": geometry_distance,
                "depth_refinement": 1.0,
                "depth_val": 0.0
            }

        # ----------------------------------------------------
        # ROBUST DEPTH VALUE
        # ----------------------------------------------------
        depth_val = float(
            np.percentile(valid, 20)
        )

        depth_val = max(
            depth_val,
            0.05
        )

        # ----------------------------------------------------
        # DEPTH CONFIDENCE
        # ----------------------------------------------------
        roi_mean = float(
            np.mean(valid)
        )

        roi_std = float(
            np.std(valid)
        )

        coeff_var = roi_std / (
            roi_mean + 1e-5
        )

        confidence = 1.0 - min(
            1.0,
            coeff_var
        )

        confidence = max(
            0.1,
            confidence
        )

        # ----------------------------------------------------
        # CONTROLLED DEPTH REFINEMENT
        # ----------------------------------------------------
        #
        # Depth Anything gives relative depth.
        # So depth must not dominate geometry.
        # ----------------------------------------------------
        raw_refinement = 1.0 / depth_val

        raw_refinement = np.sqrt(
            raw_refinement
        )

        if confidence < 0.45:

            depth_refinement = 1.0

        else:

            depth_refinement = float(
                np.clip(
                    raw_refinement,
                    0.90,
                    1.10
                )
            )

        # ----------------------------------------------------
        # FINAL DISTANCE
        # ----------------------------------------------------
        distance = (
            geometry_distance *
            depth_refinement
        )

        distance = float(
            np.clip(
                distance,
                1.0,
                100.0
            )
        )

        return {
            "distance_m": distance,
            "confidence": float(confidence),
            "geometry_distance": float(geometry_distance),
            "depth_refinement": float(depth_refinement),
            "depth_val": float(depth_val)
        }


# ============================================================
# DEPTH ESTIMATOR
# ============================================================

class DepthEstimatorDA:

    def __init__(
        self,
        model_dir: str = "models/depth_anything",
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None
    ):

        print("[INFO] Loading Depth Anything model...")

        # ----------------------------------------------------
        # DEVICE SELECTION
        # ----------------------------------------------------
        if device is None:

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        else:

            self.device = torch.device(device)

        print("[INFO] Depth device:", self.device)

        # ----------------------------------------------------
        # FP16 SELECTION
        # ----------------------------------------------------
        #
        # FP16 saves GPU memory.
        # CPU should stay FP32.
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
                f"Depth model folder not found: {model_dir}"
            )

        # ----------------------------------------------------
        # LOAD PROCESSOR
        # ----------------------------------------------------
        self.processor = AutoImageProcessor.from_pretrained(
            model_dir
        )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_dir
        )

        self.model.to(
            self.device
        )

        if self.use_fp16:

            self.model.half()

        self.model.eval()

        self.model_name = "Depth-Anything-v2"

        print(
            f"[INFO] Depth Anything loaded on {self.device} "
            f"(fp16={self.use_fp16})."
        )

    # ========================================================
    # ESTIMATE DEPTH
    # ========================================================

    def estimate(
        self,
        frame: np.ndarray
    ) -> DepthOutput:

        start = time.perf_counter()

        # ----------------------------------------------------
        # INPUT VALIDATION
        # ----------------------------------------------------
        if frame is None:

            raise ValueError(
                "Input frame is None"
            )

        if len(frame.shape) != 3:

            raise ValueError(
                "Input frame must be BGR image with 3 channels"
            )

        # ----------------------------------------------------
        # BGR TO RGB
        # ----------------------------------------------------
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------
        inputs = self.processor(
            images=rgb,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        if self.use_fp16 and "pixel_values" in inputs:

            inputs["pixel_values"] = inputs["pixel_values"].half()

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------
        with torch.inference_mode():

            outputs = self.model(
                **inputs
            )

            depth = outputs.predicted_depth

        # ----------------------------------------------------
        # RESIZE TO ORIGINAL FRAME SIZE
        # ----------------------------------------------------
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

        depth = depth.detach().float().cpu().numpy()

        # ----------------------------------------------------
        # NORMALIZE DEPTH TO 0-1
        # ----------------------------------------------------
        d_min = float(
            np.min(depth)
        )

        d_max = float(
            np.max(depth)
        )

        depth = (
            depth - d_min
        ) / (
            d_max - d_min + 1e-6
        )

        depth = depth.astype(
            np.float32
        )

        ms = (
            time.perf_counter() - start
        ) * 1000

        return DepthOutput(
            depth_map=depth,
            processing_ms=ms,
            model_name=self.model_name
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def visualize(
        self,
        frame: np.ndarray,
        depth_output: DepthOutput
    ) -> np.ndarray:

        d = (
            depth_output.depth_map * 255
        ).astype(np.uint8)

        d_col = cv2.applyColorMap(
            d,
            cv2.COLORMAP_MAGMA
        )

        if d_col.shape[:2] != frame.shape[:2]:

            d_col = cv2.resize(
                d_col,
                (frame.shape[1], frame.shape[0])
            )

        return np.hstack([
            frame,
            d_col
        ])


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

DepthEstimator = DepthEstimatorDA
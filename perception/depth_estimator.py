"""
NeuroSentinel v3 — OpenCV Depth Estimator (MiDaS ONNX)
NO TORCH, NO ADMIN REQUIRED 
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass


@dataclass
class DepthOutput:
    depth_map: np.ndarray
    uncertainty: np.ndarray
    processing_ms: float
    model_name: str

    def sample_at_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        h, w = self.depth_map.shape

        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))

        if x2 <= x1 or y2 <= y1:
            return {'distance_m': 999.0, 'confidence': 0.0}

        # bottom region = ground contact
        y_start = int(y1 + (y2 - y1) * 0.66)
        roi = self.depth_map[y_start:y2, x1:x2]

        if roi.size == 0:
            return {'distance_m': 999.0, 'confidence': 0.0}

        distance = float(np.percentile(roi, 20))
        distance = max(distance, 0.05)

        # ✅ Object-level smoothing
        key = (x1, y1, x2, y2)

        if hasattr(self, "prev_distances") and key in self.prev_distances:
            distance = 0.8 * self.prev_distances[key] + 0.2 * distance

        if hasattr(self, "prev_distances"):
            self.prev_distances[key] = distance

        # Confidence
        spread = np.std(roi)
        confidence = float(np.clip(1.0 - spread, 0.0, 1.0))

        distance_m = distance * 25 + 2

        return {
            'distance_m': round(distance_m, 1),
            'confidence': round(confidence, 2)
        }



class DepthEstimator:
    def __init__(self, model_path="models/midas_v21_384.onnx"):
        print("Loading MiDaS ONNX model...")
        self.net = cv2.dnn.readNet(model_path)
        self.model_name = "MiDaS-ONNX"
        self.prev_depth = None
        self.prev_distances = {}


    def estimate(self, frame):
        t = time.perf_counter()

        h, w = frame.shape[:2]

        # Correct preprocessing for DPT Hybrid ONNX
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (384, 384))

        img = img / 255.0
        img = (img - 0.5) / 0.5   # correct normalization

        img = img.transpose(2, 0, 1)
        blob = np.expand_dims(img, axis=0).astype(np.float32)

        # Inference
        self.net.setInput(blob)
        depth = self.net.forward()

        depth = depth[0, 0]

        # Smooth depth
        depth = cv2.GaussianBlur(depth, (5, 5), 0)

        # Resize back
        depth_resized = cv2.resize(depth, (w, h))

        # Normalize (keep RELATIVE depth, not fake meters)
        # Remove extreme spikes
        depth_clamped = np.clip(depth_resized, 0, np.percentile(depth_resized, 95))

        d_min = depth_clamped.min()
        d_max = depth_clamped.max()

        depth_norm = (depth_clamped - d_min) / (d_max - d_min + 1e-6)

        # Temporal smoothing (ADAS stability)
        if self.prev_depth is not None and self.prev_depth.shape == depth_norm.shape:
            depth_norm = 0.7 * self.prev_depth + 0.3 * depth_norm


        self.prev_depth = depth_norm.copy()

        ms = (time.perf_counter() - t) * 1000

        out = DepthOutput(
            depth_map=depth_norm.astype(np.float32),
            uncertainty=np.ones_like(depth_norm) * 0.3,
            processing_ms=ms,
            model_name=self.model_name
        )

        out.prev_distances = self.prev_distances

        if len(self.prev_distances) > 100:
            self.prev_distances.clear()


        return out

    def visualize(self, frame, depth_output):
        depth = depth_output.depth_map
        h, w = frame.shape[:2]

        
        d_vis = (depth * 255).astype(np.uint8)
        d_vis = cv2.equalizeHist(d_vis)

        d_vis = d_vis.astype(np.uint8)

        d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)

        return np.hstack([frame, d_color])

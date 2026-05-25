"""
NeuroSentinel v3 — OpenCV Depth Estimator (MiDaS ONNX)
NO TORCH, NO ADMIN REQUIRED ✅
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

        distance = float(np.percentile(roi, 15))

        return {
            'distance_m': round(distance, 1),
            'confidence': 0.8
        }


class DepthEstimator:
    def __init__(self, model_path="models/midas_small.onnx"):
        print("Loading MiDaS ONNX model...")
        self.net = cv2.dnn.readNet(model_path)
        self.model_name = "MiDaS-ONNX"

    def estimate(self, frame):
        t = time.perf_counter()

        h, w = frame.shape[:2]

        # preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

        img = img / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std
        img = img.transpose(2, 0, 1)

        blob = np.expand_dims(img, axis=0).astype(np.float32)

        self.net.setInput(blob)
        depth = self.net.forward()

        depth = depth[0, 0]

        # Smooth and enhance structure
        depth = cv2.GaussianBlur(depth, (5, 5), 0)

        # resize back
        depth_resized = cv2.resize(depth, (w, h))

        # normalize to meters (approx scale)
        d_min, d_max = depth_resized.min(), depth_resized.max()
        
        depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-6)

        # Convert to metric-like scale
        
        # Reintroduce inversion (model expects it)
        depth_metric = (1.0 - depth_norm)

        # Expand dynamic range
        depth_metric = np.power(depth_metric, 1.2) * 80.0



        ms = (time.perf_counter() - t) * 1000

        return DepthOutput(
            depth_map=depth_metric.astype(np.float32),
            uncertainty=np.ones_like(depth_metric) * 0.3,
            processing_ms=ms,
            model_name=self.model_name
        )

    def visualize(self, frame, depth_output):
        depth = depth_output.depth_map
        h, w = frame.shape[:2]

        d_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        d_vis = d_vis.astype(np.uint8)

        d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)

        return np.hstack([frame, d_color])

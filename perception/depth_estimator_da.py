import cv2
import time
import numpy as np
from dataclasses import dataclass

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


@dataclass
class DepthOutput:
    depth_map: np.ndarray
    processing_ms: float
    model_name: str
    previous_distance: float = None

    def sample_at_bbox(self, bbox):

        x1, y1, x2, y2 = map(int, bbox)

        h, w = self.depth_map.shape
        x1, x2 = max(0, x1), min(w-1, x2)
        y1, y2 = max(0, y1), min(h-1, y2)

        if x2 <= x1 or y2 <= y1:
            return {'distance_m': 999.0, 'confidence': 0.0}

        # bottom center ROI (better)
        cx = (x1 + x2) // 2
        w_box = (x2 - x1) // 4
        y_start = int(y2 - 0.2 * (y2 - y1))

        roi = self.depth_map[y_start:y2, cx-w_box:cx+w_box]

        if roi.size == 0:
            return {'distance_m': 999.0, 'confidence': 0.0}

        depth_val = float(np.percentile(roi, 20))
        depth_val = max(depth_val, 0.05)

        # better scale for depth-anything
        depth_distance = 20.0 / depth_val

        distance = depth_distance

        if self.previous_distance is not None:
            distance = 0.7 * self.previous_distance + 0.3 * distance

        self.previous_distance = distance

        confidence = 1.0 - min(1.0, roi.std() / (roi.mean() + 1e-5))

        return {
            'distance_m': float(distance),
            'confidence': float(confidence)
        }


class DepthEstimatorDA:

    def __init__(self):
        print("[INFO] Loading Depth Anything model...")

        self.device = torch.device("cpu")

        self.processor = AutoImageProcessor.from_pretrained(
            "models/depth_anything"
        )

        self.model = AutoModelForDepthEstimation.from_pretrained(
            "models/depth_anything"
        ).to(self.device)

        self.model.eval()

        self.model_name = "Depth-Anything-v2"

        print("[INFO] Depth Anything loaded.")

    def estimate(self, frame):

        start = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

        ms = (time.time() - start) * 1000

        return DepthOutput(
            depth_map=depth,
            processing_ms=ms,
            model_name=self.model_name
        )

    def visualize(self, frame, depth_output):

        d = (depth_output.depth_map * 255).astype(np.uint8)
        d_col = cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)

        return np.hstack([frame, d_col])

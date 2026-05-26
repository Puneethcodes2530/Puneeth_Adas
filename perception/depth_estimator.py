"""
NeuroSentinel v3 — OpenCV Depth Estimator (MiDaS ONNX)


This module:
✓ Runs monocular depth estimation
✓ Uses MiDaS ONNX model
✓ Gives relative depth map
✓ Estimates object distance
✓ Applies temporal smoothing
✓ Fuses geometry + depth
✓ Lightweight enough for ADAS prototype
"""

import cv2                      # OpenCV for image processing + DNN inference
import numpy as np              # Numerical operations and matrix math
import time                     # Used for measuring latency
from dataclasses import dataclass


# ============================================================
# DEPTH OUTPUT
# ============================================================
#
# Stores:
# - depth map
# - uncertainty
# - latency
# - model metadata
# ============================================================

@dataclass
class DepthOutput:

    # Relative depth map from MiDaS
    # Near objects = brighter usually
    depth_map: np.ndarray

    # Uncertainty map
    # (placeholder for future advanced logic)
    uncertainty: np.ndarray

    # Total processing time
    processing_ms: float

    # Model name
    model_name: str

    # ========================================================
    # SAMPLE DEPTH AT OBJECT BOUNDING BOX
    # ========================================================
    #
    # Takes bbox:
    # [x1, y1, x2, y2]
    #
    # Returns:
    # estimated distance + confidence
    # ========================================================
    def sample_at_bbox(self, bbox):

        # Unpack coordinates
        x1, y1, x2, y2 = bbox

        # Get depth map dimensions
        h, w = self.depth_map.shape

        # ====================================================
        # SAFETY CLAMP
        # ====================================================
        #
        # Ensures bbox never goes outside image.
        #
        # Prevents:
        # index out of bounds crash
        # ====================================================
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))

        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # ====================================================
        # INVALID BOX CHECK
        # ====================================================
        #
        # If width/height becomes invalid,
        # return fake far distance.
        # ====================================================
        if x2 <= x1 or y2 <= y1:

            return {
                'distance_m': 999.0,
                'confidence': 0.0
            }

        # ====================================================
        # OBJECT SIZE
        # ====================================================

        # Bounding box height
        bbox_h = y2 - y1

        # Bounding box width
        bbox_w = x2 - x1

        # ====================================================
        # REAL-WORLD OBJECT HEIGHTS
        # ====================================================
        #
        # Needed for pinhole camera geometry.
        #
        # Formula:
        #
        # distance =
        # focal_length * real_height
        # --------------------------
        # image_height_pixels
        # ====================================================
        REAL_HEIGHTS = {

            'person': 1.7,

            'car': 1.5,

            'truck': 3.2,

            'bus': 3.0,

            'motorcycle': 1.2,

            'bicycle': 1.1,

            'autorickshaw': 1.6
        }

        # ====================================================
        # Current object class
        # ====================================================
        #
        # AdaptiveDetector sets this externally.
        #
        # Fallback = person
        # ====================================================
        class_name = getattr(
            self,
            "current_class",
            "person"
        )

        # Real-world object height
        real_h = REAL_HEIGHTS.get(
            class_name,
            1.7
        )

        # ====================================================
        # EFFECTIVE OBJECT SIZE
        # ====================================================
        #
        # Uses weighted height + width.
        #
        # Why?
        #
        # Only height can fluctuate badly.
        #
        # Width stabilizes estimation slightly.
        # ====================================================
        effective_size = (
            0.6 * bbox_h +
            0.4 * bbox_w
        )

        # ====================================================
        # PINHOLE CAMERA MODEL
        # ====================================================
        #
        # Classical geometry estimation.
        #
        # Formula:
        #
        # H / D = h / f
        #
        # Rearranged:
        #
        # D = (f * H) / h
        #
        # H = real height
        # h = image pixel height
        # f = focal length
        # ====================================================
        FOCAL_LENGTH = 720

        heuristic_dist = (
            FOCAL_LENGTH * real_h
        ) / max(effective_size, 1)

        # ====================================================
        # DEPTH SAMPLING REGION
        # ====================================================
        #
        # Instead of whole object,
        # use bottom strip only.
        #
        # Why?
        #
        # Bottom touches road surface.
        #
        # Gives more stable depth.
        # ====================================================
        foot_y = y2

        y_start = max(
            foot_y - 10,
            0
        )

        roi = self.depth_map[
            y_start:foot_y,
            x1:x2
        ]

        # ====================================================
        # EMPTY ROI CHECK
        # ====================================================
        if roi.size == 0:

            return {
                'distance_m': 999.0,
                'confidence': 0.0
            }

        # ====================================================
        # ROBUST DEPTH VALUE
        # ====================================================
        #
        # Using percentile instead of mean.
        #
        # Why?
        #
        # Mean affected by noise/outliers.
        #
        # Percentile more stable.
        # ====================================================
        depth_val = float(
            np.percentile(roi, 20) #I am biasing towards the strongest pixels
        )

        # Prevent divide weirdness
        depth_val = max(depth_val, 0.05)

        # ====================================================
        # TEMPORAL SMOOTHING
        # ====================================================
        #
        # Stabilizes object distances across frames.
        #
        # Prevents:
        #
        # 5m → 20m → 6m ❌
        #
        # Gives:
        #
        # 5m → 6m → 6.5m ✅
        # ====================================================
        key = (x1, y1, x2, y2)

        if (
            hasattr(self, "prev_distances")
            and key in self.prev_distances
        ):

            depth_val = (
                0.8 *
                self.prev_distances[key]
                +
                0.2 * depth_val
            )

        # Store updated value
        if hasattr(self, "prev_distances"):

            self.prev_distances[key] = depth_val

        # ====================================================
        # CONFIDENCE ESTIMATION
        # ====================================================
        #
        # If depth ROI varies heavily:
        # confidence decreases.
        #
        # Stable region:
        # high confidence
        # ====================================================
        spread = np.std(roi)

        confidence = float(
            np.clip(
                1.0 - spread,
                0.0,
                1.0
            )
        )

        # ====================================================
        # DEPTH SCALING
        # ====================================================
        #
        # MiDaS gives relative depth only.
        #
        # We scale roughly into metres.
        #
        # NOT physically accurate.
        # ====================================================
        depth_dist = depth_val * 40

        # ====================================================
        # ADAPTIVE FUSION
        # ====================================================
        #
        # Combine:
        #
        # 1. Geometry distance
        # 2. Depth estimation
        #
        # If depth confidence high:
        # trust depth more
        #
        # Otherwise:
        # trust geometry more
        # ====================================================
        weight_depth = (
            0.4
            if confidence > 0.8
            else 0.2
        )

        distance_m = (

            (1 - weight_depth)
            * heuristic_dist

            +

            weight_depth
            * depth_dist
        )

        # ====================================================
        # SAFETY CLAMP
        # ====================================================
        #
        # Prevent insane values:
        #
        # negative
        # 500m
        # etc
        # ====================================================
        distance_m = np.clip(
            distance_m,
            1.0,
            80.0
        )

        # ====================================================
        # FINAL OUTPUT
        # ====================================================
        return {

            'distance_m': round(distance_m, 1),

            'confidence': round(confidence, 2)
        }


# ============================================================
# DEPTH ESTIMATOR
# ============================================================

class DepthEstimator:

    def __init__(
        self,
        model_path="models/model_fp16.onnx"
    ):

        print(
            "Loading MiDaS ONNX model..."
        )

        # ====================================================
        # Load ONNX model using OpenCV DNN
        # ====================================================
        self.net = cv2.dnn.readNet(
            "models/midas_v21_384.onnx"
        )

        self.model_name = "MiDaS-ONNX"

        # Previous depth map
        # used for temporal smoothing
        self.prev_depth = None

        # Previous object distances
        self.prev_distances = {}

    # ========================================================
    # MAIN DEPTH ESTIMATION
    # ========================================================
    def estimate(self, frame):

        # Start latency timer
        t = time.perf_counter()

        # Frame dimensions
        h, w = frame.shape[:2]

        # ====================================================
        # PREPROCESSING
        # ====================================================

        # OpenCV uses BGR
        # MiDaS trained on RGB
        img = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize to model input size
        img = cv2.resize(
            img,
            (384, 384)
        )

        # ====================================================
        # Normalize to 0-1
        # ====================================================
        img = img / 255.0

        # ====================================================
        # ImageNet normalization
        # ====================================================
        #
        # MiDaS expects ImageNet stats.
        #
        # VERY IMPORTANT.
        # ====================================================
        mean = np.array([
            0.485,
            0.456,
            0.406
        ])

        std = np.array([
            0.229,
            0.224,
            0.225
        ])

        img = (
            img - mean
        ) / std

        # ====================================================
        # Convert HWC → CHW
        # ====================================================
        #
        # OpenCV image:
        # Height Width Channels
        #
        # Model wants:
        # Channels Height Width
        # ====================================================
        img = img.transpose(
            2,
            0,
            1
        )

        # ====================================================
        # Add batch dimension
        # ====================================================
        #
        # Final shape:
        #
        # [1, 3, 384, 384]
        # ====================================================
        blob = np.expand_dims(
            img,
            axis=0
        ).astype(np.float32)

        # ====================================================
        # MODEL INFERENCE
        # ====================================================

        # Set model input
        self.net.setInput(blob)

        # Run forward pass
        depth = self.net.forward()

        # Debug print
        print(
            "RAW DEPTH SHAPE:",
            depth.shape
        )

        # Remove batch/channel dimensions
        depth = depth[0, 0]

        # ====================================================
        # DEPTH SMOOTHING
        # ====================================================
        #
        # Gaussian blur reduces:
        # noisy spikes
        # checkerboard artifacts
        # ====================================================
        depth = cv2.GaussianBlur(
            depth,
            (5, 5),
            0
        )

        # ====================================================
        # Resize back to original image size
        # ====================================================
        depth_resized = cv2.resize(
            depth,
            (w, h)
        )

        # ====================================================
        # DEPTH CLAMPING
        # ====================================================
        #
        # Removes extreme outliers.
        #
        # Prevents:
        # random huge spikes
        # ====================================================
        depth_clamped = np.clip(
            depth_resized,
            0,
            np.percentile(
                depth_resized,
                95
            )
        )

        # Min/max depth
        d_min = depth_clamped.min()
        d_max = depth_clamped.max()

        # ====================================================
        # NORMALIZATION
        # ====================================================
        #
        # Convert depth to 0-1 range.
        #
        # MiDaS outputs relative depth only.
        # ====================================================
        depth_norm = (
            depth_clamped - d_min
        ) / (
            d_max - d_min + 1e-6
        )

        # ====================================================
        # TEMPORAL SMOOTHING
        # ====================================================
        #
        # Stabilizes depth maps frame-to-frame.
        # ====================================================
        if (
            self.prev_depth is not None
            and
            self.prev_depth.shape
            ==
            depth_norm.shape
        ):

            depth_norm = (

                0.7 * self.prev_depth

                +

                0.3 * depth_norm
            )

        # Store current depth map
        self.prev_depth = depth_norm.copy()

        # ====================================================
        # TOTAL LATENCY
        # ====================================================
        ms = (
            time.perf_counter() - t
        ) * 1000

        # ====================================================
        # CREATE OUTPUT OBJECT
        # ====================================================
        out = DepthOutput(

            depth_map=depth_norm.astype(
                np.float32
            ),

            uncertainty=np.ones_like(
                depth_norm
            ) * 0.3,

            processing_ms=ms,

            model_name=self.model_name
        )

        # Share temporal memory
        out.prev_distances = (
            self.prev_distances
        )

        # ====================================================
        # MEMORY CLEANUP
        # ====================================================
        #
        # Prevent dictionary from growing forever.
        # ====================================================
        if len(self.prev_distances) > 100:

            self.prev_distances.clear()

        return out

    # ========================================================
    # DEPTH VISUALIZATION
    # ========================================================
    #
    # Creates side-by-side:
    #
    # RGB frame
    # +
    # Colored depth map
    # ========================================================
    def visualize(
        self,
        frame,
        depth_output
    ):

        depth = depth_output.depth_map

        h, w = frame.shape[:2]

        # ====================================================
        # Convert depth to grayscale image
        # ====================================================
        d_vis = (
            depth * 255
        ).astype(np.uint8)

        # Histogram equalization
        #
        # Improves depth visibility.
        d_vis = cv2.equalizeHist(
            d_vis
        )

        d_vis = d_vis.astype(
            np.uint8
        )

        # ====================================================
        # Apply color map
        # ====================================================
        #
        # Inferno:
        # near/far visually clearer
        # ====================================================
        d_color = cv2.applyColorMap(
            d_vis,
            cv2.COLORMAP_INFERNO
        )

        # ====================================================
        # Side-by-side visualization
        # ====================================================
        return np.hstack([
            frame,
            d_color
        ])

"""
NeuroSentinel v3 — Adaptive Detection Pipeline
Combines SceneDetector + YOLOv8 with dynamic thresholds.
This is Phase 2 complete pipeline.
"""
import cv2
import numpy as np
import time
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from perception.scene_detector import SceneDetector, SceneState


@dataclass
class Detection:
    class_name: str
    raw_confidence: float
    adj_confidence: float # after scene penalty
    bbox: List[int] # [x1,y1,x2,y2]
    width_px: int
    height_px: int
    est_distance_m: float # heuristic — replaced later
    is_vru: bool # pedestrian/cyclist/motorcycle


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
            'frame_id': self.frame_id,
            'timestamp_ms': self.timestamp_ms,
            'scene_condition': self.scene.condition,
            'scene_severity': self.scene.severity,
            'conf_threshold': self.scene.conf_threshold,
            'n_objects': self.n_objects,
            'processing_ms': round(self.processing_ms, 1),
            'within_budget': self.within_budget,
            'detections': [
                {
                    'class': d.class_name,
                    'raw_conf': round(d.raw_confidence, 3),
                    'adj_conf': round(d.adj_confidence, 3),
                    'bbox': d.bbox,
                    'height_px': d.height_px,
                    'est_distance_m': d.est_distance_m,
                    'is_vru': d.is_vru,
                }
                for d in self.detections
            ]
        }


class AdaptiveDetector:
    """
    Phase 2 complete detection pipeline.

    Fixed threshold (naive):
      Always use 0.35 confidence regardless of conditions.
      Result: many false positives at night/fog.

    Adaptive threshold (NeuroSentinel):
      1. Detect scene condition (2ms)
      2. Adjust confidence threshold per condition
      3. Apply confidence penalty to all detections
      4. Result: fewer false positives, more robust

    This is your first novel contribution working.
    """

    VRU_CLASSES = {
        'person', 'bicycle', 'motorcycle'}

    ADAS_CLASSES = {
        'person', 'bicycle', 'motorcycle',
        'car', 'truck', 'bus',
        'traffic light', 'stop sign'
    }

    # Heuristic distance: focal_length * real_height / pixel_height
    # Approximate real heights (metres)
    REAL_HEIGHTS = {
        'person': 1.75,
        'car': 1.50,
        'truck': 3.50,
        'bus': 3.20,
        'bicycle': 1.10,
        'motorcycle': 1.20,
    }
    FOCAL_LENGTH = 720 # approximate for dashcam

    def __init__(self, model_weights: str = 'yolov8s.pt',
                 budget_ms: float = 40.0):
        from ultralytics import YOLO
        print(f"Loading {model_weights}...")
        self.model = YOLO(model_weights)
        self.scene = SceneDetector(enforce_budget=False)
        self.budget = budget_ms
        self.frame_id = 0
        print("✓ AdaptiveDetector ready")

    def process(self, frame: np.ndarray) -> FrameOutput:
        """
        Full adaptive pipeline on one frame.
        Returns FrameOutput with all detections.
        """
        t_total = time.perf_counter()
        self.frame_id += 1

        # Step 1: Scene analysis
        scene_state = self.scene.analyze(frame)

        # Step 2: Detection with adaptive threshold
        results = self.model(
            frame,
            conf=scene_state.conf_threshold,
            verbose=False
        )
        result = results[0]

        # Step 3: Process detections
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_conf = float(box.conf[0])
            cls_name = self.model.names[int(box.cls[0])]

            if cls_name not in self.ADAS_CLASSES:
                continue

            # Apply scene penalty
            adj_conf = raw_conf * scene_state.confidence_penalty

            # Heuristic distance
            h_px = max(y2 - y1, 1)
            r_h = self.REAL_HEIGHTS.get(cls_name, 1.5)
            dist = round(self.FOCAL_LENGTH * r_h / h_px, 1)

            detections.append(Detection(
                class_name=cls_name,
                raw_confidence=round(raw_conf, 3),
                adj_confidence=round(adj_conf, 3),
                bbox=[x1, y1, x2, y2],
                width_px=x2-x1,
                height_px=h_px,
                est_distance_m=dist,
                is_vru=(cls_name in self.VRU_CLASSES)
            ))

        total_ms = (time.perf_counter() - t_total) * 1000

        return FrameOutput(
            frame_id=self.frame_id,
            timestamp_ms=time.time() * 1000,
            scene=scene_state,
            detections=detections,
            n_objects=len(detections),
            processing_ms=total_ms,
            within_budget=(total_ms <= self.budget)
        )

    def draw(self, frame: np.ndarray,
              output: FrameOutput) -> np.ndarray:
        """Draw detections + scene HUD on frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        COLORS = {
            'person': (255, 50, 50),
            'bicycle': (255, 165, 0),
            'motorcycle': (255, 140, 0),
            'car': (50, 205, 50),
            'truck': (0, 128, 255),
            'bus': (128, 0, 255),
            'traffic light': (255, 255, 0),
            'stop sign': (255, 0, 128),
        }

        COND_COLORS = {
            'CLEAR': (0, 255, 0),
            'NIGHT': (100, 100, 255),
            'FOG': (200, 200, 200),
            'RAIN': (0, 200, 255),
            'GLARE': (255, 255, 0),
            'DUST': (200, 150, 50),
        }

        # Draw detections
        for det in output.detections:
            x1, y1, x2, y2 = det.bbox
            color = COLORS.get(det.class_name, (200,200,200))
            thickness = 3 if det.is_vru else 2

            cv2.rectangle(vis, (x1,y1), (x2,y2),
                          color, thickness)

            # Label: class + adj_confidence + distance
            label = (f"{det.class_name} "
                     f"{det.adj_confidence:.2f} "
                     f"{det.est_distance_m}m")
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis,
                          (x1, y1-th-8),
                          (x1+tw+2, y1), color, -1)
            cv2.putText(vis, label, (x1+1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0,0,0), 1)

            # VRU indicator
            if det.is_vru:
                cv2.putText(vis, "VRU",
                            (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 50, 50), 2)

        # Scene condition badge (top left)
        cond = output.scene.condition
        color = COND_COLORS.get(cond, (255,255,255))
        badge = f"SCENE: {cond} ({output.scene.severity:.2f})"
        cv2.rectangle(vis, (8, 8), (300, 35),
                      (0, 0, 0), -1)
        cv2.putText(vis, badge, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

        # Threshold badge
        thresh_txt = f"CONF THRESH: {output.scene.conf_threshold}"
        cv2.rectangle(vis, (8, 38), (260, 62),
                      (0, 0, 0), -1)
        cv2.putText(vis, thresh_txt, (12, 57),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        # Objects + latency (top right)
        cv2.rectangle(vis, (w-200, 8), (w-5, 62),
                      (0, 0, 0), -1)
        cv2.putText(vis, f"Objects: {output.n_objects}",
                    (w-195, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 200), 2)
        budget_col = ((0,255,0) if output.within_budget
                      else (0,0,255))
        cv2.putText(vis,
                    f"{output.processing_ms:.0f}ms "
                    f"{'✓' if output.within_budget else '!'}",
                    (w-195, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, budget_col, 2)

        # Bottom bar
        cv2.rectangle(vis, (0, h-30), (w, h),
                      (0, 0, 0), -1)
        cv2.putText(vis,
                    f"NeuroSentinel v3 | Frame {output.frame_id} "
                    f"| YOLOv8s | Tata Technologies",
                    (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (150, 150, 150), 1)

        return vis
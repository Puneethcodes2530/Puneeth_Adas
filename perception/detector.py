"""
Adaptive Detection Pipeline
Combines SceneDetector + YOLOv8 with dynamic thresholds.
This is Phase 2 complete pipeline.
"""
import cv2 # Computer vision library used for image handling, drawing boxes and test and processing(resizing, color conversion , blur and filters)
import numpy as np #Matrix, Math operations
import time #Time measurement module
import json #Data storage in JSON format
import os #OS interface, used for file handling and directory creation
import sys #System level operations to allow imports from other files and stuff
from dataclasses import dataclass, field #A cleaner way to create data structures
from typing import List, Optional # For readability, and better use case

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
#__file__ shows the current file path and the abspath converts it into absolute path, next the os.path.dirname, moves it a level up and one more on the outside moves it even one more level up so pushing us to the root folder
#The last sys.path.insert(0, ...) ensures to include this in the python import search path, for importing further files like below in future
from perception.scene_analyzer import (
    SceneState, #CLIP imported ✅
    CLIPSceneDetector
)


@dataclass
class Detection:
    class_name: str #The detected object is stored as a string as either car, truck, person, motorcycle
    raw_confidence: float #Original yolo confidence score that the model gives
    adj_confidence: float # This is after the scene penalty (if raw confidence is 0.8 and penalty is 0.7, it becomes 0.8*0.7=0.56)
    bbox: List[int] # [x1,y1,x2,y2] for the coordinates of the bounding box, top left is (x1,y1), bottom right is (x2,y2)
    width_px: int #width of the object calculated as x2-x1
    height_px: int #height of the object is calculated as y2-y1
    est_distance_m: float # heuristic — replaced later with hybrid(depth+geometry)
    is_vru: bool # pedestrian/cyclist/motorcycle, Adas gives higher priority to VRU's


@dataclass
class FrameOutput:
    frame_id: int #Each frame gets it's own number
    timestamp_ms: float #Time when frame was processed
    scene: SceneState #Output from the Scenestate that we imported, useful like for finding if it is night and all
    detections: List[Detection] #List of all the objects in the frame
    n_objects: int #len(detections), basically for the number of the objects detected in the frame
    processing_ms: float #Time taken to process whole pipeline, Object detection+depth+scene detection
    within_budget: bool #for now capped at 40ms, but lets see regarding optimization on the later parts...

    def to_dict(self) -> dict: #Converts the frame output object to a dictionary, for easy future json output and all
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

    VRU_CLASSES = {
        'person', 'bicycle', 'motorcycle',
        'dog', 'cat', 'horse', 'cow'
    }

    ADAS_CLASSES = {
        'person', 'bicycle', 'motorcycle',
        'car', 'truck', 'bus','rickshaw'
        'traffic light', 'stop sign',
        'dog', 'cat', 'horse', 'cow'
    }

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

        self.budget = budget_ms
        self.frame_id = 0

        # ✅ CLIP initialized correctly
        self.clip = CLIPSceneDetector()

        print("The AdaptiveDetector Module is ready")

    def process(self, frame: np.ndarray) -> FrameOutput:

        t_total = time.perf_counter()
        self.frame_id += 1

        # ✅ STEP 1: CLIP Scene detection
        clip_condition, clip_conf = self.clip.analyze(frame)

        scene_state = SceneState(
            condition=clip_condition,
            severity=(1.0 - clip_conf),  # ✅ BETTER logic
            brightness=0.0,
            blur_score=0.0,
            fog_score=0.0,
            processing_ms=0.0
        )

        # ✅ STEP 2: YOLO detection
        results = self.model(
            frame,
            conf=scene_state.conf_threshold,
            verbose=False
        )

        result = results[0]

        # ✅ STEP 3: Process detections
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_conf = float(box.conf[0])
            cls_name = self.model.names[int(box.cls[0])]

            if cls_name not in self.ADAS_CLASSES:
                continue

            adj_conf = raw_conf * scene_state.confidence_penalty

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

    def draw(self, frame: np.ndarray, output: FrameOutput) -> np.ndarray:

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

        for det in output.detections:
            x1, y1, x2, y2 = det.bbox
            color = COLORS.get(det.class_name, (200,200,200))
            thickness = 3 if det.is_vru else 2

            cv2.rectangle(vis, (x1,y1), (x2,y2), color, thickness)

            label = f"{det.class_name} {det.adj_confidence:.2f} {det.est_distance_m}m"

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            cv2.rectangle(vis, (x1, y1-th-8), (x1+tw+2, y1), color, -1)

            cv2.putText(vis, label, (x1+1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

            if det.is_vru:
                cv2.putText(vis, "VRU",
                            (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 50, 50), 2)

        cond = output.scene.condition
        color = COND_COLORS.get(cond, (255,255,255))

        badge = f"SCENE: {cond} ({output.scene.severity:.2f})"

        cv2.rectangle(vis, (8, 8), (300, 35), (0, 0, 0), -1)

        cv2.putText(vis, badge, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        thresh_txt = f"CONF THRESH: {output.scene.conf_threshold}"

        cv2.rectangle(vis, (8, 38), (260, 62), (0, 0, 0), -1)

        cv2.putText(vis, thresh_txt, (12, 57),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        cv2.rectangle(vis, (w-200, 8), (w-5, 62), (0, 0, 0), -1)

        cv2.putText(vis, f"Objects: {output.n_objects}",
                    (w-195, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 200), 2)

        budget_col = (0,255,0) if output.within_budget else (0,0,255)

        cv2.putText(vis,
                    f"{output.processing_ms:.0f}ms {'✓' if output.within_budget else '!'}",
                    (w-195, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, budget_col, 2)

        cv2.rectangle(vis, (0, h-30), (w, h), (0, 0, 0), -1)

        cv2.putText(vis,
                    f"NeuroSentinel v3 | Frame {output.frame_id} | YOLOv8s | Tata Technologies",
                    (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (150, 150, 150), 1)

        return vis
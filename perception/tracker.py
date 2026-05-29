"""
NeuroSentinel v3 — Phase 4 Tracker Pipeline

Features:
✓ YOLO tracking / detection
✓ Persistent IDs when tracker is available
✓ Geometry-first distance estimation
✓ Depth-assisted refinement
✓ TTC estimation using:
    - distance change
    - bbox height growth / tau-margin
✓ Weighted risk scoring
✓ FCW / AEB / VRU-ready output
✓ CPU-friendly configurable settings
"""

import cv2
import numpy as np
import time

from ultralytics import YOLO
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TrackState:
    track_id: int
    class_name: str
    bbox: List[int]

    age: int = 0
    missed: int = 0

    bbox_heights: deque = field(
        default_factory=lambda: deque(maxlen=10)
    )

    distances: deque = field(
        default_factory=lambda: deque(maxlen=10)
    )


@dataclass
class TTCResult:
    value: Optional[float]
    zone: str


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    confidence: float
    bbox: List[int]
    distance_m: float
    ttc: TTCResult
    risk: str
    risk_score: float


# ============================================================
# TTC ENGINE
# ============================================================

class TTCEngine:
    """
    TTC is estimated using two complementary methods:

    1. Distance velocity:
       If distance is reducing over time, estimate TTC.

    2. Tau-margin using bbox height growth:
       If object bbox height is increasing, object is approaching.

    Final TTC is selected/fused based on availability and stability.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / max(fps, 1e-6)

    def _zone(self, ttc: float) -> str:
        if ttc < 1.5:
            return "CRITICAL"
        if ttc < 3.0:
            return "WARNING"
        if ttc < 6.0:
            return "CAUTION"
        return "SAFE"

    def compute(self, state: TrackState) -> TTCResult:

        if len(state.distances) < 3:
            return TTCResult(
                value=None,
                zone="SAFE"
            )

        # ----------------------------------------------------
        # PATH A: Distance-velocity TTC
        # ----------------------------------------------------
        d = list(state.distances)

        velocities = []

        for i in range(1, len(d)):

            # Positive velocity means object is approaching.
            v = (d[i - 1] - d[i]) / self.dt

            velocities.append(v)

        v_mean = float(np.mean(velocities))
        v_std = float(np.std(velocities)) + 1.0

        ttc_a = 999.0

        if v_mean > 0:

            ttc_a = float(
                np.clip(
                    d[-1] / (v_mean + 1e-6),
                    0.1,
                    99.0
                )
            )

        # ----------------------------------------------------
        # PATH B: Tau-margin from bbox height growth
        # ----------------------------------------------------
        ttc_b = 999.0
        sig_b = 999.0

        if len(state.bbox_heights) >= 3:

            heights = list(state.bbox_heights)
            taus = []

            for i in range(1, len(heights)):

                dh = (
                    heights[i] -
                    heights[i - 1]
                ) * self.fps

                if dh > 0.3 and heights[i] > 0:

                    tau = heights[i] / dh

                    if 0.5 < tau < 60.0:
                        taus.append(tau)

            if len(taus) > 0:

                ttc_b = float(
                    np.median(taus)
                )

                sig_b = (
                    float(np.std(taus)) +
                    0.5
                )

        # ----------------------------------------------------
        # ARBITRATION
        # ----------------------------------------------------
        if ttc_a >= 99.0 and ttc_b >= 99.0:

            return TTCResult(
                value=None,
                zone="SAFE"
            )

        elif ttc_a >= 99.0:

            ttc_final = ttc_b

        elif ttc_b >= 99.0:

            ttc_final = ttc_a

        else:

            # Inverse variance weighting.
            w_a = 1.0 / (
                v_std ** 2 + 1e-6
            )

            w_b = 1.0 / (
                sig_b ** 2 + 1e-6
            )

            ttc_final = (
                w_a * ttc_a +
                w_b * ttc_b
            ) / (
                w_a + w_b
            )

        return TTCResult(
            value=round(float(ttc_final), 2),
            zone=self._zone(float(ttc_final))
        )


# ============================================================
# PHASE 4 TRACKER PIPELINE
# ============================================================

class Phase4TrackerPipeline:

    REAL_HEIGHTS = {
        "person": 1.7,
        "car": 1.5,
        "truck": 3.5,
        "bus": 3.2,
        "motorcycle": 1.2,
        "bicycle": 1.2,
        "autorickshaw": 1.6
    }

    CLASS_WEIGHTS = {
        "person": 1.0,
        "bicycle": 0.9,
        "motorcycle": 0.85,
        "car": 0.7,
        "truck": 0.75,
        "bus": 0.75,
        "autorickshaw": 0.8
    }

    VRU_CLASSES = {
        "person",
        "bicycle",
        "motorcycle"
    }

    YOLO_CLASSES = [
        0,   # person
        1,   # bicycle
        2,   # car
        3,   # motorcycle
        5,   # bus
        7,   # truck
        9,   # traffic light
        11   # stop sign
    ]

    def __init__(
        self,
        model_weights: str = "yolov8s.pt",
        depth_fn=None,
        fps: float = 30.0,
        focal_length: float = 721.5,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        use_tracker: bool = True
    ):

        print("=" * 60)
        print("Loading NeuroSentinel v3 Phase 4 Tracker Pipeline")
        print("=" * 60)

        self.model_weights = model_weights
        self.model = YOLO(model_weights)

        print(f"✓ YOLO loaded: {model_weights}")

        self.depth_fn = depth_fn

        self.fps = fps
        self.focal_length = focal_length

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.use_tracker = use_tracker

        self.ttc_engine = TTCEngine(
            fps=fps
        )

        self.tracks: Dict[int, TrackState] = {}

        self.frame_id = 0

        print("✓ Pipeline ready")

    # ========================================================
    # IMAGE ENHANCEMENT
    # ========================================================

    def enhance_frame(
        self,
        frame: np.ndarray
    ) -> np.ndarray:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        brightness = float(
            np.mean(gray)
        )

        enhanced = frame.copy()

        # ----------------------------------------------------
        # Night enhancement
        # ----------------------------------------------------
        if brightness < 75:

            hsv = cv2.cvtColor(
                enhanced,
                cv2.COLOR_BGR2HSV
            )

            h, s, v = cv2.split(hsv)

            v = cv2.equalizeHist(v)

            hsv = cv2.merge(
                [h, s, v]
            )

            enhanced = cv2.cvtColor(
                hsv,
                cv2.COLOR_HSV2BGR
            )

        # ----------------------------------------------------
        # CLAHE contrast enhancement
        # ----------------------------------------------------
        lab = cv2.cvtColor(
            enhanced,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge(
            (l, a, b)
        )

        enhanced = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

        return enhanced

    # ========================================================
    # DISTANCE ESTIMATION
    # ========================================================

    def estimate_distance(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        bbox: List[int],
        class_name: str
    ) -> float:

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        H, W = frame.shape[:2]

        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))

        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        if x2 <= x1 or y2 <= y1:
            return 999.0

        # ----------------------------------------------------
        # GEOMETRY DISTANCE
        # ----------------------------------------------------
        bbox_h = max(
            y2 - y1,
            1
        )

        bbox_w = max(
            x2 - x1,
            1
        )

        real_h = self.REAL_HEIGHTS.get(
            class_name,
            1.5
        )

        effective_size = (
            0.75 * bbox_h +
            0.25 * bbox_w
        )

        geo_distance = (
            self.focal_length * real_h
        ) / max(
            effective_size,
            1
        )

        geo_distance = float(
            np.clip(
                geo_distance,
                1.0,
                100.0
            )
        )

        # ----------------------------------------------------
        # If no depth, return geometry
        # ----------------------------------------------------
        if depth_map is None:
            return geo_distance

        if depth_map.shape[:2] != (H, W):

            depth_map = cv2.resize(
                depth_map,
                (W, H)
            )

        # ----------------------------------------------------
        # BOTTOM-CENTER ROI
        # ----------------------------------------------------
        cx = (x1 + x2) // 2

        half_w = max(
            4,
            bbox_w // 6
        )

        y_start = int(
            y2 - 0.20 * bbox_h
        )

        roi = depth_map[
            y_start:y2,
            max(0, cx - half_w):min(W, cx + half_w)
        ]

        if roi.size == 0:
            return geo_distance

        valid = roi[
            np.isfinite(roi)
        ]

        valid = valid[
            valid > 0
        ]

        if valid.size == 0:
            return geo_distance

        # ----------------------------------------------------
        # DEPTH VALUE + CONFIDENCE
        # ----------------------------------------------------
        depth_val = float(
            np.percentile(valid, 20)
        )

        depth_val = max(
            depth_val,
            0.05
        )

        roi_mean = float(
            np.mean(valid)
        )

        roi_std = float(
            np.std(valid)
        )

        coeff_var = roi_std / (
            roi_mean + 1e-5
        )

        depth_conf = 1.0 - min(
            1.0,
            coeff_var
        )

        # ----------------------------------------------------
        # DEPTH DISTANCE HEURISTIC
        # ----------------------------------------------------
        #
        # Depth Anything is relative, not metric.
        # This is only a weak correction signal.
        # ----------------------------------------------------
        depth_distance = (
            depth_val * 22.0
        ) + 3.0

        depth_distance = float(
            np.clip(
                depth_distance,
                1.0,
                100.0
            )
        )

        # ----------------------------------------------------
        # SMART GEOMETRY-FIRST FUSION
        # ----------------------------------------------------
        if depth_conf < 0.45:

            fused = geo_distance

        else:

            if geo_distance < 10:

                fused = (
                    0.80 * geo_distance +
                    0.20 * depth_distance
                )

            elif geo_distance < 30:

                fused = (
                    0.72 * geo_distance +
                    0.28 * depth_distance
                )

            else:

                fused = (
                    0.85 * geo_distance +
                    0.15 * depth_distance
                )

        fused = float(
            np.clip(
                fused,
                1.0,
                100.0
            )
        )

        return fused

    # ========================================================
    # RISK SCORING
    # ========================================================

    def compute_risk(
        self,
        class_name: str,
        ttc: TTCResult,
        distance_m: float,
        bbox: List[int],
        frame_w: int
    ) -> Tuple[float, str]:

        # ----------------------------------------------------
        # TTC component: 45%
        # ----------------------------------------------------
        if ttc.value is None:

            ttc_s = 0.05

        elif ttc.value < 1.5:

            ttc_s = 1.0

        elif ttc.value < 3.0:

            ttc_s = 0.8

        elif ttc.value < 6.0:

            ttc_s = 0.5

        else:

            ttc_s = 0.1

        # ----------------------------------------------------
        # Class component: 25%
        # ----------------------------------------------------
        cls_s = self.CLASS_WEIGHTS.get(
            class_name,
            0.5
        )

        if class_name in self.VRU_CLASSES:

            cls_s = max(
                cls_s,
                0.85
            )

        # ----------------------------------------------------
        # Lane component: 20%
        # ----------------------------------------------------
        cx = (
            bbox[0] +
            bbox[2]
        ) // 2

        in_ego_lane = (
            0.35 * frame_w <
            cx <
            0.65 * frame_w
        )

        lane_s = (
            1.0
            if in_ego_lane
            else 0.3
        )

        # ----------------------------------------------------
        # Distance component: 10%
        # ----------------------------------------------------
        if distance_m < 10:

            dist_s = 1.0

        elif distance_m < 30:

            dist_s = 0.6

        else:

            dist_s = 0.2

        # ----------------------------------------------------
        # Weighted final score
        # ----------------------------------------------------
        score = (
            0.45 * ttc_s +
            0.25 * cls_s +
            0.20 * lane_s +
            0.10 * dist_s
        )

        if score > 0.80:

            level = "CRITICAL"

        elif score > 0.60:

            level = "HIGH"

        elif score > 0.40:

            level = "MEDIUM"

        else:

            level = "LOW"

        return float(score), level

    # ========================================================
    # TRACK CLEANUP
    # ========================================================

    def _cleanup_tracks(
        self,
        active_ids
    ):

        for tid in list(self.tracks.keys()):

            if tid not in active_ids:

                self.tracks[tid].missed += 1

                if self.tracks[tid].missed > 30:

                    del self.tracks[tid]

            else:

                self.tracks[tid].missed = 0

    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(
        self,
        frame: np.ndarray
    ):

        self.frame_id += 1

        start = time.perf_counter()

        H, W = frame.shape[:2]

        infer_frame = self.enhance_frame(
            frame
        )

        # ----------------------------------------------------
        # DEPTH
        # ----------------------------------------------------
        depth_map = None

        if self.depth_fn is not None:

            try:

                depth_map = self.depth_fn(
                    infer_frame
                )

            except Exception as e:

                print(
                    f"[WARNING] Depth failed: {e}"
                )

                depth_map = None

        # ----------------------------------------------------
        # YOLO TRACKING / DETECTION
        # ----------------------------------------------------
        try:

            if self.use_tracker:

                results = self.model.track(
                    infer_frame,
                    persist=True,
                    verbose=False,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    agnostic_nms=True,
                    tracker="bytetrack.yaml",
                    classes=self.YOLO_CLASSES
                )

            else:

                results = self.model(
                    infer_frame,
                    verbose=False,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    agnostic_nms=True,
                    classes=self.YOLO_CLASSES
                )

        except Exception as e:

            print(
                f"[WARNING] Tracker failed, falling back to detection: {e}"
            )

            results = self.model(
                infer_frame,
                verbose=False,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                agnostic_nms=True,
                classes=self.YOLO_CLASSES
            )

        result = results[0]

        objects = []

        active_ids = set()

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(len(boxes)):

                x1, y1, x2, y2 = map(
                    int,
                    boxes.xyxy[i].tolist()
                )

                conf = float(
                    boxes.conf[i].item()
                )

                cls_id = int(
                    boxes.cls[i].item()
                )

                cls_name = self.model.names[
                    cls_id
                ]

                # ------------------------------------------------
                # TRACK ID
                # ------------------------------------------------
                if hasattr(boxes, "id") and boxes.id is not None:

                    tid = int(
                        boxes.id[i].item()
                    )

                else:

                    tid = i + self.frame_id * 10000

                active_ids.add(tid)

                # ------------------------------------------------
                # TRACK STATE
                # ------------------------------------------------
                if tid not in self.tracks:

                    self.tracks[tid] = TrackState(
                        track_id=tid,
                        class_name=cls_name,
                        bbox=[x1, y1, x2, y2]
                    )

                state = self.tracks[tid]

                state.age += 1
                state.class_name = cls_name
                state.bbox = [x1, y1, x2, y2]

                bbox_h = max(
                    y2 - y1,
                    1
                )

                state.bbox_heights.append(
                    bbox_h
                )

                # ------------------------------------------------
                # DISTANCE
                # ------------------------------------------------
                distance = self.estimate_distance(
                    frame,
                    depth_map,
                    [x1, y1, x2, y2],
                    cls_name
                )

                # Light distance smoothing per track
                if len(state.distances) > 0:

                    previous = state.distances[-1]

                    distance = (
                        0.65 * previous +
                        0.35 * distance
                    )

                state.distances.append(
                    distance
                )

                # ------------------------------------------------
                # TTC
                # ------------------------------------------------
                ttc = self.ttc_engine.compute(
                    state
                )

                # ------------------------------------------------
                # WEIGHTED RISK
                # ------------------------------------------------
                risk_score, risk = self.compute_risk(
                    class_name=cls_name,
                    ttc=ttc,
                    distance_m=distance,
                    bbox=[x1, y1, x2, y2],
                    frame_w=W
                )

                objects.append(
                    TrackedObject(
                        track_id=tid,
                        class_name=cls_name,
                        confidence=round(conf, 2),
                        bbox=[x1, y1, x2, y2],
                        distance_m=round(float(distance), 1),
                        ttc=ttc,
                        risk=risk,
                        risk_score=round(risk_score, 3)
                    )
                )

        self._cleanup_tracks(
            active_ids
        )

        latency = (
            time.perf_counter() -
            start
        ) * 1000

        return objects, depth_map, latency

    # ========================================================
    # DRAWING
    # ========================================================

    def draw(
        self,
        frame: np.ndarray,
        objects: List[TrackedObject],
        latency: float
    ) -> np.ndarray:

        vis = frame.copy()

        colors = {
            "LOW": (0, 255, 0),
            "MEDIUM": (0, 200, 255),
            "HIGH": (0, 120, 255),
            "CRITICAL": (0, 0, 255)
        }

        for obj in objects:

            x1, y1, x2, y2 = obj.bbox

            color = colors.get(
                obj.risk,
                (255, 255, 255)
            )

            thickness = (
                3
                if obj.risk == "CRITICAL"
                else 2
            )

            cv2.rectangle(
                vis,
                (x1, y1),
                (x2, y2),
                color,
                thickness
            )

            ttc_text = (
                f"{obj.ttc.value:.2f}s"
                if obj.ttc.value is not None
                else "N/A"
            )

            lines = [
                f"ID:{obj.track_id}",
                f"{obj.class_name} {obj.confidence:.2f}",
                f"{obj.distance_m:.1f}m",
                f"TTC:{ttc_text}",
                f"Risk:{obj.risk}",
                f"Score:{obj.risk_score:.2f}"
            ]

            for idx, line in enumerate(lines):

                yy = y1 - 85 + idx * 15

                if yy < 8:

                    yy = y2 + 15 + idx * 15

                cv2.putText(
                    vis,
                    line,
                    (x1, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA
                )

        # ----------------------------------------------------
        # HUD
        # ----------------------------------------------------
        cv2.rectangle(
            vis,
            (0, 0),
            (vis.shape[1], 58),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            vis,
            (
                f"NeuroSentinel v3 | {self.model_weights} | "
                f"Tracking + Distance + TTC + Risk | "
                f"{latency:.0f} ms"
            ),
            (10, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        return vis
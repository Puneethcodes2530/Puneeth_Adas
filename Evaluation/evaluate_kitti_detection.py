"""
NeuroSentinel v3 — KITTI Evaluation
Geometry vs Geometry+Depth Fusion

Evaluates:
1. Detection performance
2. Geometry-only distance estimation
3. Geometry + depth fusion estimation

Outputs:
- MAE
- RMSE
- AbsRel
- Improvement %
- Scatter plots
- JSON report
"""

import cv2
import numpy as np
import os
import sys
import glob
import json
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict


# ============================================================
# IMPORT PROJECT
# ============================================================

sys.path.insert(
    0,
    os.path.abspath(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )
)

from perception.detector import AdaptiveDetector
from perception.depth_estimator import DepthEstimatorDA


# ============================================================
# CONFIG
# ============================================================

KITTI_ROOT = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI"

MAX_IMAGES = 100

IOU_THRESHOLD = 0.5


# ============================================================
# KITTI LABEL
# ============================================================

@dataclass
class KITTILabel:

    obj_type: str
    bbox_2d: list
    z_3d: float
    truncated: float
    occluded: int

    @property
    def adas_class(self):

        mapping = {

            'Car': 'car',

            'Van': 'car',

            'Truck': 'truck',

            'Pedestrian': 'person',

            'Person_sitting': 'person',

            'Cyclist': 'bicycle',

            'Tram': 'bus'
        }

        return mapping.get(
            self.obj_type,
            None
        )

    @property
    def is_valid(self):

        return (

            self.truncated < 0.5 and

            self.occluded < 2 and

            self.z_3d > 0 and

            self.z_3d < 80
        )


# ============================================================
# PARSE LABELS
# ============================================================

def parse_kitti_label_file(label_path):

    labels = []

    with open(label_path, 'r') as f:

        for line in f.readlines():

            parts = line.strip().split()

            if len(parts) < 15:
                continue

            obj_type = parts[0]

            if obj_type == "DontCare":
                continue

            labels.append(

                KITTILabel(

                    obj_type=obj_type,

                    truncated=float(parts[1]),

                    occluded=int(parts[2]),

                    bbox_2d=[

                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7])
                    ],

                    z_3d=float(parts[13])
                )
            )

    return labels


# ============================================================
# IOU
# ============================================================

def compute_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    if inter <= 0:
        return 0.0

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = area1 + area2 - inter

    return inter / (union + 1e-6)


# ============================================================
# GEOMETRY DISTANCE
# ============================================================

def geometry_distance(det):

    x1, y1, x2, y2 = det.bbox

    bbox_h = y2 - y1
    bbox_w = x2 - x1

    effective_size = (

        0.7 * bbox_h +

        0.3 * bbox_w
    )

    REAL_HEIGHTS = {

        'person': 1.7,

        'car': 1.5,

        'truck': 3.2,

        'bus': 3.0,

        'motorcycle': 1.2,

        'bicycle': 1.1
    }

    real_h = REAL_HEIGHTS.get(
        det.class_name,
        1.7
    )

    FOCAL_LENGTH = 850

    dist = (
        FOCAL_LENGTH * real_h
    ) / max(effective_size, 1)

    return dist


# ============================================================
# DEPTH FUSION
# ============================================================

def fusion_distance(det, depth_out):

    geom_dist = geometry_distance(det)

    x1, y1, x2, y2 = map(int, det.bbox)

    h, w = depth_out.depth_map.shape

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))

    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return geom_dist

    # --------------------------------------------------------
    # FOOT ROI
    # --------------------------------------------------------

    foot_y = y2

    y_start = max(
        foot_y - 12,
        0
    )

    roi = depth_out.depth_map[
        y_start:foot_y,
        x1:x2
    ]

    if roi.size == 0:
        return geom_dist

    # --------------------------------------------------------
    # DEPTH STABILITY
    # --------------------------------------------------------

    depth_mean = float(
        np.mean(roi)
    )

    depth_std = float(
        np.std(roi)
    )

    stability = 1.0 - min(
        1.0,
        depth_std / (depth_mean + 1e-6)
    )

    # --------------------------------------------------------
    # RELATIVE DEPTH
    # --------------------------------------------------------

    raw_depth = float(
        np.percentile(roi, 20)
    )

    raw_depth = max(raw_depth, 0.05)

    # --------------------------------------------------------
    # SMALL CORRECTION
    # --------------------------------------------------------

    correction = 1.0 + (
        (0.5 - raw_depth) * 0.25
    )

    correction = np.clip(
        correction,
        0.85,
        1.15
    )

    depth_adjusted = (
        geom_dist * correction
    )

    # --------------------------------------------------------
    # CONFIDENCE WEIGHTED FUSION
    # --------------------------------------------------------

    fusion_weight = 0.15 * stability

    final_dist = (

        (1 - fusion_weight) * geom_dist +

        fusion_weight * depth_adjusted
    )

    final_dist = np.clip(
        final_dist,
        1.0,
        120.0
    )

    return float(final_dist)


# ============================================================
# DISTANCE METRICS
# ============================================================

def compute_distance_metrics(gt, pred):

    gt = np.array(gt)
    pred = np.array(pred)

    if len(gt) == 0:
        return {}

    # scale alignment
    scale = np.median(gt) / (
        np.median(pred) + 1e-6
    )

    pred_scaled = pred * scale

    mae = np.mean(
        np.abs(pred_scaled - gt)
    )

    rmse = np.sqrt(
        np.mean((pred_scaled - gt) ** 2)
    )

    abs_rel = np.mean(
        np.abs(pred_scaled - gt) / gt
    )

    return {

        'scale_factor': round(scale, 3),

        'MAE_m': round(float(mae), 3),

        'RMSE_m': round(float(rmse), 3),

        'AbsRel': round(float(abs_rel), 3)
    }


# ============================================================
# MAIN
# ============================================================

def run_evaluation():

    print("=" * 60)
    print("NeuroSentinel v3 — KITTI Evaluation")
    print("=" * 60)

    image_dir = os.path.join(

        KITTI_ROOT,

        "data_object_image_2",

        "training",

        "image_2"
    )

    label_dir = os.path.join(

        KITTI_ROOT,

        "data_object_label_2",

        "training",

        "label_2"
    )

    image_files = sorted(
        glob.glob(
            os.path.join(image_dir, "*.png")
        )
    )[:MAX_IMAGES]

    print("Loading models...")

    detector = AdaptiveDetector(
        'yolov8s.pt'
    )

    depth_estimator = DepthEstimatorDA()

    print("✓ Models loaded")

    # ========================================================
    # STORAGE
    # ========================================================

    all_gt_dist = []

    all_geom_dist = []

    all_fusion_dist = []

    latencies = []

    class_results = defaultdict(
        lambda: {

            'tp': 0,

            'fp': 0,

            'fn': 0
        }
    )

    print(f"Evaluating on {len(image_files)} images")

    # ========================================================
    # LOOP
    # ========================================================

    for idx, image_path in enumerate(image_files):

        image_id = os.path.basename(
            image_path
        ).replace(".png", "")

        label_path = os.path.join(

            label_dir,

            image_id + ".txt"
        )

        frame = cv2.imread(image_path)

        gt_labels = [

            l for l in
            parse_kitti_label_file(label_path)

            if l.is_valid and l.adas_class
        ]

        # ----------------------------------------------------
        # PIPELINE
        # ----------------------------------------------------

        start = time.time()

        det_output = detector.process(frame)

        depth_out = depth_estimator.estimate(frame)

        latency = (
            time.time() - start
        ) * 1000

        latencies.append(latency)

        # ----------------------------------------------------
        # MATCHING
        # ----------------------------------------------------

        matched_pred = set()

        for gt in gt_labels:

            best_iou = IOU_THRESHOLD
            best_det = None

            for det_idx, det in enumerate(
                det_output.detections
            ):

                if det_idx in matched_pred:
                    continue

                if det.class_name != gt.adas_class:
                    continue

                iou = compute_iou(
                    gt.bbox_2d,
                    det.bbox
                )

                if iou > best_iou:

                    best_iou = iou
                    best_det = det_idx

            # ------------------------------------------------
            # MATCH FOUND
            # ------------------------------------------------

            if best_det is not None:

                matched_pred.add(best_det)

                det = det_output.detections[
                    best_det
                ]

                gt_dist = gt.z_3d

                geom_dist = geometry_distance(
                    det
                )

                fusion_dist = fusion_distance(
                    det,
                    depth_out
                )

                # --------------------------------------------
                # STORE
                # --------------------------------------------

                all_gt_dist.append(gt_dist)

                all_geom_dist.append(geom_dist)

                all_fusion_dist.append(fusion_dist)

                class_results[
                    det.class_name
                ]['tp'] += 1

            else:

                class_results[
                    gt.adas_class
                ]['fn'] += 1

        # ----------------------------------------------------
        # FALSE POSITIVES
        # ----------------------------------------------------

        for det_idx, det in enumerate(
            det_output.detections
        ):

            if det_idx not in matched_pred:

                class_results[
                    det.class_name
                ]['fp'] += 1

        print(

            f"[{idx+1}/{len(image_files)}] "

            f"{image_id} "

            f"| GT:{len(gt_labels)} "

            f"| Pred:{len(det_output.detections)} "

            f"| Latency:{latency:.0f}ms"
        )

    # ========================================================
    # METRICS
    # ========================================================

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    print("Detection Metrics:")

    for cls, r in class_results.items():

        tp = r['tp']
        fp = r['fp']
        fn = r['fn']

        precision = tp / (tp + fp + 1e-6)

        recall = tp / (tp + fn + 1e-6)

        f1 = (

            2 * precision * recall
        ) / (
            precision + recall + 1e-6
        )

        print(

            f"{cls:<15}"

            f"P:{precision:.3f} "

            f"R:{recall:.3f} "

            f"F1:{f1:.3f}"
        )

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    geom_metrics = compute_distance_metrics(

        all_gt_dist,

        all_geom_dist
    )

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    fusion_metrics = compute_distance_metrics(

        all_gt_dist,

        all_fusion_dist
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DISTANCE ESTIMATION COMPARISON")
    print("=" * 60)

    print("\nGeometry Only:")

    for k, v in geom_metrics.items():

        print(f"{k:<15}: {v}")

    print("\nGeometry + Depth Fusion:")

    for k, v in fusion_metrics.items():

        print(f"{k:<15}: {v}")

    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    geom_mae = geom_metrics['MAE_m']

    fusion_mae = fusion_metrics['MAE_m']

    improvement = (

        (geom_mae - fusion_mae)

        / geom_mae

    ) * 100

    print("\n" + "=" * 60)

    print(
        f"Fusion improved MAE by "
        f"{improvement:.2f}%"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    mean_latency = np.mean(
        latencies
    )

    fps = 1000 / mean_latency

    print("\nLatency:")

    print(
        f"Mean latency: "
        f"{mean_latency:.1f}ms"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    os.makedirs(
        "outputs/evaluation",
        exist_ok=True
    )

    results = {

        "geometry_metrics":
            geom_metrics,

        "fusion_metrics":
            fusion_metrics,

        "fusion_improvement_percent":
            round(improvement, 2),

        "mean_latency_ms":
            float(mean_latency),

        "fps":
            float(fps)
    }

    with open(

        "outputs/evaluation/results.json",

        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # ========================================================
    # SCATTER PLOTS
    # ========================================================

    gt = np.array(all_gt_dist)

    geom = np.array(all_geom_dist)

    fusion = np.array(all_fusion_dist)

    geom_scale = np.median(gt) / (
        np.median(geom) + 1e-6
    )

    fusion_scale = np.median(gt) / (
        np.median(fusion) + 1e-6
    )

    geom = geom * geom_scale

    fusion = fusion * fusion_scale

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    axes[0].scatter(
        gt,
        geom,
        alpha=0.6
    )

    axes[0].plot(
        [0, max(gt)],
        [0, max(gt)],
        'r--'
    )

    axes[0].set_title(
        "Geometry Only"
    )

    axes[0].set_xlabel(
        "GT Distance (m)"
    )

    axes[0].set_ylabel(
        "Predicted Distance (m)"
    )

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    axes[1].scatter(
        gt,
        fusion,
        alpha=0.6
    )

    axes[1].plot(
        [0, max(gt)],
        [0, max(gt)],
        'r--'
    )

    axes[1].set_title(
        "Geometry + Depth Fusion"
    )

    axes[1].set_xlabel(
        "GT Distance (m)"
    )

    axes[1].set_ylabel(
        "Predicted Distance (m)"
    )

    plt.tight_layout()

    plt.savefig(

        "outputs/evaluation/comparison.png",

        dpi=150
    )

    plt.show()

    print("\n✓ Evaluation Complete")

    print("Saved to:")

    print("outputs/evaluation/")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    run_evaluation()
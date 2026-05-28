"""
NeuroSentinel v3
KITTI Comparison Evaluation

Compares:

1. Geometry-only distance estimation
2. Geometry + Depth Fusion estimation

Outputs:
- MAE
- RMSE
- AbsRel
- % improvement
- comparison plots
"""

import cv2
import numpy as np
import os
import sys
import glob
import json
import time

import matplotlib.pyplot as plt

from collections import defaultdict
from dataclasses import dataclass


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

MAX_IMAGES = 5

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
# METRICS
# ============================================================

def compute_metrics(gt, pred):

    gt = np.array(gt)
    pred = np.array(pred)

    scale = np.median(gt) / (
        np.median(pred) + 1e-6
    )

    pred = pred * scale

    mae = np.mean(
        np.abs(pred - gt)
    )

    rmse = np.sqrt(
        np.mean((pred - gt) ** 2)
    )

    abs_rel = np.mean(
        np.abs(pred - gt) / gt
    )

    return {

        'scale': scale,

        'MAE': mae,

        'RMSE': rmse,

        'AbsRel': abs_rel
    }, pred


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
# FUSION DISTANCE
# ============================================================

def fusion_distance(det, depth_out):

    geom_dist = geometry_distance(det)

    x1, y1, x2, y2 = map(int, det.bbox)

    foot_y = y2

    y_start = max(
        foot_y - 10,
        0
    )

    roi = depth_out.depth_map[
        y_start:foot_y,
        x1:x2
    ]

    if roi.size == 0:
        return geom_dist

    raw = float(
        np.percentile(roi, 20)
    )

    raw = max(raw, 0.05)

    depth_refine = 1.0 / raw

    depth_refine = np.clip(
        depth_refine,
        0.7,
        1.3
    )

    fused = (
        geom_dist *
        depth_refine
    )

    return fused


# ============================================================
# MAIN
# ============================================================

def run():

    print("=" * 60)
    print("KITTI Geometry vs Fusion Evaluation")
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

    print("\nLoading models...")

    detector = AdaptiveDetector(
        'yolov8s.pt'
    )

    depth_estimator = DepthEstimatorDA()

    print("✓ Models loaded")

    image_files = sorted(
        glob.glob(
            os.path.join(
                image_dir,
                "*.png"
            )
        )
    )

    image_files = image_files[:MAX_IMAGES]

    gt_all = []

    geom_all = []

    fusion_all = []

    for idx, img_path in enumerate(image_files):

        img_name = os.path.splitext(
            os.path.basename(img_path)
        )[0]

        label_path = os.path.join(
            label_dir,
            img_name + ".txt"
        )

        frame = cv2.imread(img_path)

        if frame is None:
            continue

        gt_labels = [

            l for l in
            parse_kitti_label_file(label_path)

            if l.is_valid and l.adas_class
        ]

        det_output = detector.process(frame)

        depth_out = depth_estimator.estimate(frame)

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

                gt_all.append(gt_dist)

                geom_all.append(geom_dist)

                fusion_all.append(fusion_dist)

        print(
            f"[{idx+1}/{len(image_files)}] "
            f"{img_name}"
        )

    # ========================================================
    # METRICS
    # ========================================================

    geom_metrics, geom_scaled = compute_metrics(
        gt_all,
        geom_all
    )

    fusion_metrics, fusion_scaled = compute_metrics(
        gt_all,
        fusion_all
    )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    mae_improve = (

        (
            geom_metrics['MAE'] -

            fusion_metrics['MAE']
        )

        /

        geom_metrics['MAE']
    ) * 100

    rmse_improve = (

        (
            geom_metrics['RMSE'] -

            fusion_metrics['RMSE']
        )

        /

        geom_metrics['RMSE']
    ) * 100

    absrel_improve = (

        (
            geom_metrics['AbsRel'] -

            fusion_metrics['AbsRel']
        )

        /

        geom_metrics['AbsRel']
    ) * 100

    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print("\nGEOMETRY ONLY")

    print(
        f"MAE : {geom_metrics['MAE']:.3f}m"
    )

    print(
        f"RMSE : {geom_metrics['RMSE']:.3f}m"
    )

    print(
        f"AbsRel : {geom_metrics['AbsRel']:.3f}"
    )

    print("\nDEPTH FUSION")

    print(
        f"MAE : {fusion_metrics['MAE']:.3f}m"
    )

    print(
        f"RMSE : {fusion_metrics['RMSE']:.3f}m"
    )

    print(
        f"AbsRel : {fusion_metrics['AbsRel']:.3f}"
    )

    print("\nIMPROVEMENT")

    print(
        f"MAE Improvement : "
        f"{mae_improve:.2f}%"
    )

    print(
        f"RMSE Improvement : "
        f"{rmse_improve:.2f}%"
    )

    print(
        f"AbsRel Improvement : "
        f"{absrel_improve:.2f}%"
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        "outputs/comparison",
        exist_ok=True
    )

    report = {

        'geometry': {

            'MAE': float(
                geom_metrics['MAE']
            ),

            'RMSE': float(
                geom_metrics['RMSE']
            ),

            'AbsRel': float(
                geom_metrics['AbsRel']
            )
        },

        'fusion': {

            'MAE': float(
                fusion_metrics['MAE']
            ),

            'RMSE': float(
                fusion_metrics['RMSE']
            ),

            'AbsRel': float(
                fusion_metrics['AbsRel']
            )
        },

        'improvement_percent': {

            'MAE': float(mae_improve),

            'RMSE': float(rmse_improve),

            'AbsRel': float(absrel_improve)
        }
    }

    with open(

        "outputs/comparison/"
        "comparison_results.json",

        'w'
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ========================================================
    # PLOTS
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7)
    )

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    axes[0].scatter(
        gt_all,
        geom_scaled,
        alpha=0.5
    )

    max_d = max(gt_all)

    axes[0].plot(
        [0, max_d],
        [0, max_d],
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
        gt_all,
        fusion_scaled,
        alpha=0.5
    )

    axes[1].plot(
        [0, max_d],
        [0, max_d],
        'r--'
    )

    axes[1].set_title(
        "Depth Fusion"
    )

    axes[1].set_xlabel(
        "GT Distance (m)"
    )

    axes[1].set_ylabel(
        "Predicted Distance (m)"
    )

    plt.tight_layout()

    plt.savefig(

        "outputs/comparison/"
        "comparison_plot.png",

        dpi=150
    )

    plt.show()

    print(
        "\nSaved:"
        "\noutputs/comparison/"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    run()
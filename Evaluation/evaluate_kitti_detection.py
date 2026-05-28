"""
NeuroSentinel v3 — KITTI Detection + Distance Evaluation
"""

import cv2
import numpy as np
import os
import sys
import glob
import json
import csv
import time

import matplotlib.pyplot as plt

from collections import defaultdict
from dataclasses import dataclass


# ============================================================
# PROJECT IMPORTS
# ============================================================

sys.path.insert(
    0,
    os.path.abspath(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )
)

from perception.adaptive_detector import AdaptiveDetector
from perception.depth_estimator_da import DepthEstimatorDA


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
# PARSE KITTI LABEL FILE
# ============================================================

def parse_kitti_label_file(label_path):

    labels = []

    if not os.path.exists(label_path):

        return labels

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
# DISTANCE METRICS
# ============================================================

def compute_metrics(gt, pred):

    gt = np.array(gt)
    pred = np.array(pred)

    if len(gt) == 0:

        return None

    # Scale alignment
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

        'scale_factor': round(scale, 3),

        'MAE_m': round(float(mae), 3),

        'RMSE_m': round(float(rmse), 3),

        'AbsRel': round(float(abs_rel), 3)
    }


# ============================================================
# MAIN
# ============================================================

def run():

    print("=" * 60)
    print("NeuroSentinel v3 — KITTI Evaluation")
    print("=" * 60)

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print("\nLoading models...")

    detector = AdaptiveDetector(
        'yolov8s.pt'
    )

    depth_estimator = DepthEstimatorDA()

    print("✓ Models loaded")

    # --------------------------------------------------------
    # GET IMAGES
    # --------------------------------------------------------

    image_files = sorted(

        glob.glob(
            os.path.join(
                image_dir,
                "*.png"
            )
        )
    )

    image_files = image_files[:MAX_IMAGES]

    print(f"\nEvaluating on {len(image_files)} images")

    # --------------------------------------------------------
    # RESULTS STORAGE
    # --------------------------------------------------------

    gt_all = []
    pred_all = []

    class_stats = defaultdict(
        lambda: {
            'tp': 0,
            'fp': 0,
            'fn': 0
        }
    )

    latencies = []

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

        t0 = time.time()

        det_output = detector.process(frame)

        latency_ms = (
            time.time() - t0
        ) * 1000

        latencies.append(latency_ms)

        # ----------------------------------------------------
        # DEPTH
        # ----------------------------------------------------

        depth_out = depth_estimator.estimate(frame)

        # ----------------------------------------------------
        # MATCH GT ↔ PRED
        # ----------------------------------------------------

        matched_gt = set()
        matched_pred = set()

        for gt_idx, gt in enumerate(gt_labels):

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

                matched_gt.add(gt_idx)
                matched_pred.add(best_det)

                det = det_output.detections[best_det]

                # --------------------------------------------
                # DISTANCE ESTIMATION
                # --------------------------------------------

                depth_out.current_class = det.class_name

                sample = depth_out.sample_at_bbox(
                    det.bbox
                )

                pred_dist = sample['distance_m']

                gt_dist = gt.z_3d

                gt_all.append(gt_dist)
                pred_all.append(pred_dist)

                class_stats[gt.adas_class]['tp'] += 1

            else:

                class_stats[gt.adas_class]['fn'] += 1

        # ----------------------------------------------------
        # FALSE POSITIVES
        # ----------------------------------------------------

        for det_idx, det in enumerate(
            det_output.detections
        ):

            if det_idx not in matched_pred:

                class_stats[det.class_name]['fp'] += 1

        print(

            f"[{idx+1}/{len(image_files)}] "

            f"{img_name} | "

            f"GT:{len(gt_labels)} | "

            f"Pred:{len(det_output.detections)} | "

            f"Latency:{latency_ms:.0f}ms"
        )

    # ========================================================
    # METRICS
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    # --------------------------------------------------------
    # DETECTION METRICS
    # --------------------------------------------------------

    print("\nDetection Metrics:\n")

    for cls, stats in class_stats.items():

        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']

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
    # DISTANCE METRICS
    # --------------------------------------------------------

    metrics = compute_metrics(
        gt_all,
        pred_all
    )

    print("\nDistance Metrics:\n")

    if metrics:

        for k, v in metrics.items():

            print(f"{k:<15}: {v}")

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    print("\nLatency:\n")

    print(

        f"Mean latency: "

        f"{np.mean(latencies):.1f}ms"
    )

    print(

        f"FPS: "

        f"{1000/np.mean(latencies):.2f}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        "outputs/evaluation",
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    report = {

        'distance_metrics': metrics,

        'mean_latency_ms': float(
            np.mean(latencies)
        ),

        'fps': float(
            1000 / np.mean(latencies)
        )
    }

    with open(

        "outputs/evaluation/kitti_results.json",

        'w'
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # SCATTER PLOT
    # --------------------------------------------------------

    if len(gt_all) > 0:

        gt = np.array(gt_all)
        pred = np.array(pred_all)

        scale = np.median(gt) / (
            np.median(pred) + 1e-6
        )

        pred = pred * scale

        plt.figure(figsize=(8, 8))

        plt.scatter(
            gt,
            pred,
            alpha=0.5
        )

        max_d = max(
            gt.max(),
            pred.max()
        )

        plt.plot(

            [0, max_d],

            [0, max_d],

            'r--'
        )

        plt.xlabel("GT Distance (m)")
        plt.ylabel("Predicted Distance (m)")

        plt.title(
            "KITTI Distance Evaluation"
        )

        plt.grid(True)

        plt.savefig(

            "outputs/evaluation/"
            "distance_scatter.png",

            dpi=150
        )

        plt.close()

    print("\n✓ Evaluation Complete")

    print(
        "\nSaved to:"
        "\noutputs/evaluation/"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    run()
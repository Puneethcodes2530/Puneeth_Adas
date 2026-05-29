"""
NeuroSentinel v3 — KITTI Evaluation

Evaluates:
1. Detection performance
2. Geometry-only distance estimation
3. Geometry + Depth Fusion estimation

Outputs:
- Precision / Recall / F1
- MAE
- RMSE
- AbsRel
- Accuracy %
- Improvement %
- Scatter plots
- JSON report

How to run:
python Evaluation/evaluate_kitti_detection.py
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
# PROJECT ROOT
# ============================================================

ROOT = os.path.abspath(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

sys.path.insert(
    0,
    ROOT
)

print("Project root:", ROOT)


# ============================================================
# IMPORT PROJECT
# ============================================================

from perception.detector import AdaptiveDetector
from perception.depth_estimator import DepthEstimatorDA


# ============================================================
# CONFIG — CHANGE ONLY THESE IF NEEDED
# ============================================================

KITTI_ROOT = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI"

MAX_IMAGES = 100

IOU_THRESHOLD = 0.5

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs",
    "evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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
            "Car": "car",
            "Van": "car",
            "Truck": "truck",
            "Pedestrian": "person",
            "Person_sitting": "person",
            "Cyclist": "bicycle",
            "Tram": "bus"
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
# PARSE KITTI LABELS
# ============================================================

def parse_kitti_label_file(label_path):

    labels = []

    if not os.path.exists(label_path):

        return labels

    with open(label_path, "r") as f:

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

    bbox_h = max(y2 - y1, 1)
    bbox_w = max(x2 - x1, 1)

    effective_size = (
        0.75 * bbox_h +
        0.25 * bbox_w
    )

    real_heights = {
        "person": 1.7,
        "car": 1.5,
        "truck": 3.2,
        "bus": 3.0,
        "motorcycle": 1.2,
        "bicycle": 1.1
    }

    real_h = real_heights.get(
        det.class_name,
        1.5
    )

    focal_length = 850.0

    dist = (
        focal_length *
        real_h
    ) / max(effective_size, 1)

    return float(
        np.clip(
            dist,
            1.0,
            120.0
        )
    )


# ============================================================
# DEPTH FUSION DISTANCE
# ============================================================

def fusion_distance(det, depth_out):

    """
    Uses robust sample_at_bbox() from updated depth_estimator.py.
    Falls back to geometry if anything fails.
    """

    try:

        sample = depth_out.sample_at_bbox(
            det.bbox,
            class_name=det.class_name
        )

        return float(
            sample["distance_m"]
        )

    except Exception as e:

        print(
            f"[WARNING] Fusion failed for {det.class_name}: {e}"
        )

        return geometry_distance(det)


# ============================================================
# DISTANCE METRICS
# ============================================================

def compute_distance_metrics(gt, pred, apply_scale=True):

    gt = np.array(
        gt,
        dtype=np.float32
    )

    pred = np.array(
        pred,
        dtype=np.float32
    )

    if len(gt) == 0 or len(pred) == 0:

        return {
            "scale_factor": 1.0,
            "MAE_m": 0.0,
            "RMSE_m": 0.0,
            "AbsRel": 0.0,
            "AccuracyPercent": 0.0
        }, pred

    if apply_scale:

        scale = np.median(gt) / (
            np.median(pred) + 1e-6
        )

        pred_scaled = pred * scale

    else:

        scale = 1.0
        pred_scaled = pred

    mae = np.mean(
        np.abs(pred_scaled - gt)
    )

    rmse = np.sqrt(
        np.mean((pred_scaled - gt) ** 2)
    )

    abs_rel = np.mean(
        np.abs(pred_scaled - gt) / (gt + 1e-6)
    )

    accuracy = (
        1.0 - abs_rel
    ) * 100.0

    return {
        "scale_factor": round(float(scale), 3),
        "MAE_m": round(float(mae), 3),
        "RMSE_m": round(float(rmse), 3),
        "AbsRel": round(float(abs_rel), 3),
        "AccuracyPercent": round(float(accuracy), 2)
    }, pred_scaled


# ============================================================
# DETECTION METRICS
# ============================================================

def compute_detection_summary(class_results):

    summary = {}

    for cls, r in class_results.items():

        tp = r["tp"]
        fp = r["fp"]
        fn = r["fn"]

        precision = tp / (
            tp + fp + 1e-6
        )

        recall = tp / (
            tp + fn + 1e-6
        )

        f1 = (
            2 * precision * recall
        ) / (
            precision + recall + 1e-6
        )

        summary[cls] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f1": round(float(f1), 3)
        }

    return summary


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation():

    print("=" * 70)
    print("NeuroSentinel v3 — KITTI Evaluation")
    print("=" * 70)

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

    print("Image dir:", image_dir)
    print("Label dir:", label_dir)

    if not os.path.exists(image_dir):

        print("[ERROR] KITTI image directory not found.")
        return

    if not os.path.exists(label_dir):

        print("[ERROR] KITTI label directory not found.")
        return

    image_files = sorted(
        glob.glob(
            os.path.join(
                image_dir,
                "*.png"
            )
        )
    )[:MAX_IMAGES]

    if len(image_files) == 0:

        print("[ERROR] No KITTI images found.")
        return

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading models...")

    detector = AdaptiveDetector(
        model_weights="yolov8s.pt",
        budget_ms=1000.0,
        scene_update_every=15
    )

    depth_estimator = DepthEstimatorDA()

    print("✓ Models loaded")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_gt_dist = []
    all_geom_dist = []
    all_fusion_dist = []

    latencies = []
    detection_latencies = []
    depth_latencies = []

    class_results = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    )

    image_summaries = []

    print(f"\nEvaluating on {len(image_files)} images")

    # ========================================================
    # LOOP
    # ========================================================

    for idx, image_path in enumerate(image_files):

        image_id = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        label_path = os.path.join(
            label_dir,
            image_id + ".txt"
        )

        frame = cv2.imread(
            image_path
        )

        if frame is None:
            continue

        gt_labels = [
            label
            for label in parse_kitti_label_file(label_path)
            if label.is_valid and label.adas_class
        ]

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        start_total = time.perf_counter()

        start_det = time.perf_counter()

        det_output = detector.process(
            frame
        )

        det_ms = (
            time.perf_counter() -
            start_det
        ) * 1000

        detection_latencies.append(
            det_ms
        )

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------

        depth_out = depth_estimator.estimate(
            frame
        )

        depth_ms = depth_out.processing_ms

        depth_latencies.append(
            depth_ms
        )

        latency = (
            time.perf_counter() -
            start_total
        ) * 1000

        latencies.append(
            latency
        )

        # ----------------------------------------------------
        # Matching
        # ----------------------------------------------------

        matched_pred = set()
        matches = 0

        for gt in gt_labels:

            best_iou = IOU_THRESHOLD
            best_det_idx = None

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
                    best_det_idx = det_idx

            # ------------------------------------------------
            # Match found
            # ------------------------------------------------

            if best_det_idx is not None:

                matched_pred.add(
                    best_det_idx
                )

                det = det_output.detections[
                    best_det_idx
                ]

                gt_dist = gt.z_3d

                geom_dist = geometry_distance(
                    det
                )

                fusion_dist = fusion_distance(
                    det,
                    depth_out
                )

                all_gt_dist.append(
                    gt_dist
                )

                all_geom_dist.append(
                    geom_dist
                )

                all_fusion_dist.append(
                    fusion_dist
                )

                class_results[
                    det.class_name
                ]["tp"] += 1

                matches += 1

            else:

                class_results[
                    gt.adas_class
                ]["fn"] += 1

        # ----------------------------------------------------
        # False positives
        # ----------------------------------------------------

        for det_idx, det in enumerate(
            det_output.detections
        ):

            if det_idx not in matched_pred:

                class_results[
                    det.class_name
                ]["fp"] += 1

        image_summaries.append(
            {
                "image_id": image_id,
                "gt_objects": len(gt_labels),
                "detections": len(det_output.detections),
                "matches": matches,
                "latency_ms": round(float(latency), 1),
                "detection_ms": round(float(det_ms), 1),
                "depth_ms": round(float(depth_ms), 1)
            }
        )

        print(
            f"[{idx + 1:03d}/{len(image_files)}] "
            f"{image_id} | "
            f"GT:{len(gt_labels):<2} "
            f"Pred:{len(det_output.detections):<2} "
            f"Match:{matches:<2} "
            f"Det:{det_ms:.0f}ms "
            f"Depth:{depth_ms:.0f}ms "
            f"Total:{latency:.0f}ms"
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    detection_summary = compute_detection_summary(
        class_results
    )

    print("\nDetection Metrics:")

    for cls, s in detection_summary.items():

        print(
            f"{cls:<15} "
            f"P:{s['precision']:.3f} "
            f"R:{s['recall']:.3f} "
            f"F1:{s['f1']:.3f} "
            f"TP:{s['tp']} "
            f"FP:{s['fp']} "
            f"FN:{s['fn']}"
        )

    # --------------------------------------------------------
    # Distance metrics
    # --------------------------------------------------------

    if len(all_gt_dist) == 0:

        print(
            "\n[ERROR] No matched objects found. "
            "Try IOU_THRESHOLD = 0.3 or increase MAX_IMAGES."
        )

        return

    geom_metrics, geom_scaled = compute_distance_metrics(
        all_gt_dist,
        all_geom_dist,
        apply_scale=True
    )

    fusion_metrics, fusion_scaled = compute_distance_metrics(
        all_gt_dist,
        all_fusion_dist,
        apply_scale=True
    )

    geom_mae = geom_metrics["MAE_m"]
    fusion_mae = fusion_metrics["MAE_m"]

    improvement = (
        (geom_mae - fusion_mae) /
        (geom_mae + 1e-6)
    ) * 100

    print("\n" + "=" * 70)
    print("DISTANCE ESTIMATION COMPARISON")
    print("=" * 70)

    print("\nGeometry Only:")

    for k, v in geom_metrics.items():

        print(
            f"{k:<18}: {v}"
        )

    print("\nGeometry + Depth Fusion:")

    for k, v in fusion_metrics.items():

        print(
            f"{k:<18}: {v}"
        )

    print("\n" + "=" * 70)
    print(
        f"Fusion improved MAE by {improvement:.2f}%"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    mean_latency = float(
        np.mean(latencies)
    )

    fps = float(
        1000.0 / mean_latency
    )

    print("\nLatency:")
    print(f"Mean total latency : {mean_latency:.1f} ms")
    print(f"Mean detection     : {np.mean(detection_latencies):.1f} ms")
    print(f"Mean depth         : {np.mean(depth_latencies):.1f} ms")
    print(f"FPS                : {fps:.2f}")

    # ========================================================
    # SAVE JSON
    # ========================================================

    results = {
        "config": {
            "KITTI_ROOT": KITTI_ROOT,
            "MAX_IMAGES": MAX_IMAGES,
            "IOU_THRESHOLD": IOU_THRESHOLD,
            "detector": "AdaptiveDetector + YOLOv8s",
            "depth_model": "Depth-Anything-v2"
        },
        "detection_metrics": detection_summary,
        "geometry_metrics": geom_metrics,
        "fusion_metrics": fusion_metrics,
        "fusion_improvement_percent": round(float(improvement), 2),
        "latency": {
            "mean_total_ms": mean_latency,
            "mean_detection_ms": float(np.mean(detection_latencies)),
            "mean_depth_ms": float(np.mean(depth_latencies)),
            "fps": fps
        },
        "counts": {
            "images_evaluated": len(image_files),
            "matched_objects": len(all_gt_dist)
        },
        "image_summaries": image_summaries
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        "results.json"
    )

    with open(
        json_path,
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

    gt = np.array(
        all_gt_dist
    )

    geom = np.array(
        geom_scaled
    )

    fusion = np.array(
        fusion_scaled
    )

    max_d = max(
        float(np.max(gt)),
        float(np.max(geom)),
        float(np.max(fusion))
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 6)
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    axes[0].scatter(
        gt,
        geom,
        alpha=0.6
    )

    axes[0].plot(
        [0, max_d],
        [0, max_d],
        "r--"
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

    axes[0].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    axes[1].scatter(
        gt,
        fusion,
        alpha=0.6
    )

    axes[1].plot(
        [0, max_d],
        [0, max_d],
        "r--"
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

    axes[1].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Geometry vs Fusion
    # --------------------------------------------------------

    axes[2].scatter(
        geom,
        fusion,
        alpha=0.6
    )

    axes[2].plot(
        [0, max_d],
        [0, max_d],
        "g--"
    )

    axes[2].set_title(
        "Geometry vs Fusion"
    )

    axes[2].set_xlabel(
        "Geometry Prediction (m)"
    )

    axes[2].set_ylabel(
        "Fusion Prediction (m)"
    )

    axes[2].grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        "comparison.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print("\n✓ Evaluation Complete")
    print("Saved to:")
    print(json_path)
    print(plot_path)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    run_evaluation()
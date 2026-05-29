"""
NeuroSentinel v3 — KITTI Geometry vs Depth Fusion Evaluation

Compares:
1. Geometry-only distance estimation
2. Geometry + Depth Fusion estimation

Outputs:
- MAE
- RMSE
- AbsRel
- improvement percentage
- GT vs Geometry plot
- GT vs Fusion plot
- Geometry vs Fusion plot
- Fusion difference histogram
- JSON report

How to use:
Change MAX_IMAGES / KITTI_ROOT if needed and run:

python Evaluation/evaluate_kitti_comparison.py
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
    "comparison"
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
# PARSE LABELS
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
# METRICS
# ============================================================

def compute_metrics(gt, pred, apply_scale=True):

    gt = np.array(gt, dtype=np.float32)
    pred = np.array(pred, dtype=np.float32)

    if len(gt) == 0 or len(pred) == 0:

        return {
            "scale": 1.0,
            "MAE": 0.0,
            "RMSE": 0.0,
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

    accuracy_percent = (
        1.0 - abs_rel
    ) * 100.0

    return {
        "scale": float(scale),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "AbsRel": float(abs_rel),
        "AccuracyPercent": float(accuracy_percent)
    }, pred_scaled


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
# FUSION DISTANCE
# ============================================================

def fusion_distance(det, depth_out):

    """
    Uses robust DepthOutput.sample_at_bbox() from updated depth_estimator.py.
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
# MATCH DETECTIONS TO GT
# ============================================================

def match_detections(gt_labels, detections):

    matches = []

    matched_pred = set()

    for gt in gt_labels:

        best_iou = IOU_THRESHOLD
        best_det_idx = None

        for det_idx, det in enumerate(detections):

            if det_idx in matched_pred:
                continue

            if det.class_name != gt.adas_class:
                continue

            score = compute_iou(
                gt.bbox_2d,
                det.bbox
            )

            if score > best_iou:

                best_iou = score
                best_det_idx = det_idx

        if best_det_idx is not None:

            matched_pred.add(
                best_det_idx
            )

            matches.append(
                (
                    gt,
                    detections[best_det_idx],
                    best_iou
                )
            )

    return matches


# ============================================================
# MAIN
# ============================================================

def run():

    print("=" * 70)
    print("KITTI Geometry vs Depth Fusion Evaluation")
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

    image_files = sorted(
        glob.glob(
            os.path.join(
                image_dir,
                "*.png"
            )
        )
    )

    image_files = image_files[:MAX_IMAGES]

    if len(image_files) == 0:

        print("[ERROR] No KITTI images found.")
        return

    print(f"\nEvaluating {len(image_files)} images...")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    gt_all = []
    geom_all = []
    fusion_all = []

    class_stats = defaultdict(
        lambda: {
            "count": 0,
            "geom_err": [],
            "fusion_err": []
        }
    )

    image_summaries = []

    total_depth_ms = []
    total_detector_ms = []
    total_matches = 0

    start_all = time.perf_counter()

    # --------------------------------------------------------
    # Loop images
    # --------------------------------------------------------

    for idx, img_path in enumerate(image_files):

        img_name = os.path.splitext(
            os.path.basename(img_path)
        )[0]

        label_path = os.path.join(
            label_dir,
            img_name + ".txt"
        )

        frame = cv2.imread(
            img_path
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

        det_start = time.perf_counter()

        det_output = detector.process(
            frame
        )

        det_ms = (
            time.perf_counter() -
            det_start
        ) * 1000

        total_detector_ms.append(
            det_ms
        )

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------

        depth_out = depth_estimator.estimate(
            frame
        )

        total_depth_ms.append(
            depth_out.processing_ms
        )

        # ----------------------------------------------------
        # Match GT and detections
        # ----------------------------------------------------

        matches = match_detections(
            gt_labels,
            det_output.detections
        )

        total_matches += len(matches)

        for gt, det, iou_score in matches:

            gt_dist = gt.z_3d

            geom_dist = geometry_distance(
                det
            )

            fusion_dist = fusion_distance(
                det,
                depth_out
            )

            gt_all.append(
                gt_dist
            )

            geom_all.append(
                geom_dist
            )

            fusion_all.append(
                fusion_dist
            )

            geom_err = abs(
                geom_dist - gt_dist
            )

            fusion_err = abs(
                fusion_dist - gt_dist
            )

            cls = det.class_name

            class_stats[cls]["count"] += 1
            class_stats[cls]["geom_err"].append(geom_err)
            class_stats[cls]["fusion_err"].append(fusion_err)

        image_summaries.append(
            {
                "image_id": img_name,
                "valid_gt": len(gt_labels),
                "detections": len(det_output.detections),
                "matches": len(matches),
                "detector_ms": round(det_ms, 1),
                "depth_ms": round(depth_out.processing_ms, 1)
            }
        )

        print(
            f"[{idx + 1:03d}/{len(image_files)}] "
            f"{img_name} | "
            f"GT:{len(gt_labels):<2} "
            f"Det:{len(det_output.detections):<2} "
            f"Match:{len(matches):<2} "
            f"Det:{det_ms:.0f}ms "
            f"Depth:{depth_out.processing_ms:.0f}ms"
        )

    total_time_ms = (
        time.perf_counter() -
        start_all
    ) * 1000

    # ========================================================
    # METRICS
    # ========================================================

    if len(gt_all) == 0:

        print(
            "\n[ERROR] No matched objects found. "
            "Try lowering IOU_THRESHOLD to 0.3 or increase MAX_IMAGES."
        )

        return

    geom_metrics, geom_scaled = compute_metrics(
        gt_all,
        geom_all,
        apply_scale=True
    )

    fusion_metrics, fusion_scaled = compute_metrics(
        gt_all,
        fusion_all,
        apply_scale=True
    )

    mae_improve = (
        (
            geom_metrics["MAE"] -
            fusion_metrics["MAE"]
        )
        /
        (
            geom_metrics["MAE"] + 1e-6
        )
    ) * 100

    rmse_improve = (
        (
            geom_metrics["RMSE"] -
            fusion_metrics["RMSE"]
        )
        /
        (
            geom_metrics["RMSE"] + 1e-6
        )
    ) * 100

    absrel_improve = (
        (
            geom_metrics["AbsRel"] -
            fusion_metrics["AbsRel"]
        )
        /
        (
            geom_metrics["AbsRel"] + 1e-6
        )
    ) * 100

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\nImages evaluated: {len(image_files)}")
    print(f"Matched objects:  {len(gt_all)}")
    print(f"Total runtime:    {total_time_ms:.1f} ms")
    print(f"Mean detector:    {np.mean(total_detector_ms):.1f} ms")
    print(f"Mean depth:       {np.mean(total_depth_ms):.1f} ms")

    print("\nGEOMETRY ONLY")
    print(f"Scale factor : {geom_metrics['scale']:.3f}")
    print(f"MAE          : {geom_metrics['MAE']:.3f} m")
    print(f"RMSE         : {geom_metrics['RMSE']:.3f} m")
    print(f"AbsRel       : {geom_metrics['AbsRel']:.3f}")
    print(f"Accuracy     : {geom_metrics['AccuracyPercent']:.2f}%")

    print("\nGEOMETRY + DEPTH FUSION")
    print(f"Scale factor : {fusion_metrics['scale']:.3f}")
    print(f"MAE          : {fusion_metrics['MAE']:.3f} m")
    print(f"RMSE         : {fusion_metrics['RMSE']:.3f} m")
    print(f"AbsRel       : {fusion_metrics['AbsRel']:.3f}")
    print(f"Accuracy     : {fusion_metrics['AccuracyPercent']:.2f}%")

    print("\nIMPROVEMENT")
    print(f"MAE Improvement    : {mae_improve:.2f}%")
    print(f"RMSE Improvement   : {rmse_improve:.2f}%")
    print(f"AbsRel Improvement : {absrel_improve:.2f}%")

    print("\nCLASS-WISE ERROR")
    print("-" * 70)

    for cls, s in class_stats.items():

        geom_mae_cls = np.mean(
            s["geom_err"]
        )

        fusion_mae_cls = np.mean(
            s["fusion_err"]
        )

        cls_improve = (
            (geom_mae_cls - fusion_mae_cls) /
            (geom_mae_cls + 1e-6)
        ) * 100

        print(
            f"{cls:<12} "
            f"N:{s['count']:<4} "
            f"G-MAE:{geom_mae_cls:>6.2f} "
            f"F-MAE:{fusion_mae_cls:>6.2f} "
            f"Imp:{cls_improve:>7.2f}%"
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    report = {
        "config": {
            "KITTI_ROOT": KITTI_ROOT,
            "MAX_IMAGES": MAX_IMAGES,
            "IOU_THRESHOLD": IOU_THRESHOLD,
            "model": "yolov8s.pt",
            "depth_model": "Depth-Anything-v2"
        },
        "summary": {
            "images_evaluated": len(image_files),
            "matched_objects": len(gt_all),
            "total_runtime_ms": float(total_time_ms),
            "mean_detector_ms": float(np.mean(total_detector_ms)),
            "mean_depth_ms": float(np.mean(total_depth_ms))
        },
        "geometry": {
            "scale": geom_metrics["scale"],
            "MAE": geom_metrics["MAE"],
            "RMSE": geom_metrics["RMSE"],
            "AbsRel": geom_metrics["AbsRel"],
            "AccuracyPercent": geom_metrics["AccuracyPercent"]
        },
        "fusion": {
            "scale": fusion_metrics["scale"],
            "MAE": fusion_metrics["MAE"],
            "RMSE": fusion_metrics["RMSE"],
            "AbsRel": fusion_metrics["AbsRel"],
            "AccuracyPercent": fusion_metrics["AccuracyPercent"]
        },
        "improvement_percent": {
            "MAE": float(mae_improve),
            "RMSE": float(rmse_improve),
            "AbsRel": float(absrel_improve)
        },
        "class_stats": {
            cls: {
                "count": int(s["count"]),
                "geometry_mae": float(np.mean(s["geom_err"])),
                "fusion_mae": float(np.mean(s["fusion_err"]))
            }
            for cls, s in class_stats.items()
        },
        "image_summaries": image_summaries
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        "comparison_results.json"
    )

    with open(
        json_path,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ========================================================
    # PLOTS
    # ========================================================

    gt_arr = np.array(
        gt_all
    )

    geom_scaled = np.array(
        geom_scaled
    )

    fusion_scaled = np.array(
        fusion_scaled
    )

    max_d = max(
        float(np.max(gt_arr)),
        float(np.max(geom_scaled)),
        float(np.max(fusion_scaled))
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 12)
    )

    # --------------------------------------------------------
    # GT vs Geometry
    # --------------------------------------------------------

    axes[0, 0].scatter(
        gt_arr,
        geom_scaled,
        alpha=0.55
    )

    axes[0, 0].plot(
        [0, max_d],
        [0, max_d],
        "r--"
    )

    axes[0, 0].set_title(
        "GT vs Geometry Only"
    )

    axes[0, 0].set_xlabel(
        "GT Distance (m)"
    )

    axes[0, 0].set_ylabel(
        "Geometry Prediction (m)"
    )

    axes[0, 0].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # GT vs Fusion
    # --------------------------------------------------------

    axes[0, 1].scatter(
        gt_arr,
        fusion_scaled,
        alpha=0.55
    )

    axes[0, 1].plot(
        [0, max_d],
        [0, max_d],
        "r--"
    )

    axes[0, 1].set_title(
        "GT vs Geometry + Depth Fusion"
    )

    axes[0, 1].set_xlabel(
        "GT Distance (m)"
    )

    axes[0, 1].set_ylabel(
        "Fusion Prediction (m)"
    )

    axes[0, 1].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Geometry vs Fusion
    # --------------------------------------------------------

    axes[1, 0].scatter(
        geom_scaled,
        fusion_scaled,
        alpha=0.55
    )

    axes[1, 0].plot(
        [0, max_d],
        [0, max_d],
        "g--"
    )

    axes[1, 0].set_title(
        "Geometry vs Fusion"
    )

    axes[1, 0].set_xlabel(
        "Geometry Prediction (m)"
    )

    axes[1, 0].set_ylabel(
        "Fusion Prediction (m)"
    )

    axes[1, 0].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Difference histogram
    # --------------------------------------------------------

    diff = fusion_scaled - geom_scaled

    axes[1, 1].hist(
        diff,
        bins=30
    )

    axes[1, 1].set_title(
        "Fusion - Geometry Difference"
    )

    axes[1, 1].set_xlabel(
        "Distance Difference (m)"
    )

    axes[1, 1].set_ylabel(
        "Count"
    )

    axes[1, 1].grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        "comparison_plot.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print("\nSaved:")
    print(json_path)
    print(plot_path)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    run()

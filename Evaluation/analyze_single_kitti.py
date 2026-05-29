"""
NeuroSentinel v3 — Single KITTI Image Analysis

Compares:
1. Ground Truth Distance
2. Geometry-only Distance
3. Geometry + Depth Fusion Distance

For ONE KITTI training image.

Outputs:
- visual comparison
- per-object errors
- depth map
- saved analysis image

How to use:
Change IMAGE_ID below and run:
python Evaluation/analyze_single_kitti.py
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass


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
# IMPORT PROJECT MODULES
# ============================================================

from perception.detector import AdaptiveDetector
from perception.depth_estimator import DepthEstimatorDA


# ============================================================
# CONFIG — CHANGE ONLY THESE IF NEEDED
# ============================================================

KITTI_ROOT = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI"

# CHANGE THIS IMAGE ID ONLY
IMAGE_ID = "007479"

IOU_THRESHOLD = 0.5

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs",
    "single_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# KITTI LABEL STRUCTURE
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
# LABEL PARSER
# ============================================================

def parse_kitti_label_file(label_path):

    labels = []

    if not os.path.exists(label_path):

        print("[ERROR] Label file not found:", label_path)
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
    Uses the robust depth sampling function from DepthOutput.

    This is safer than manually duplicating depth fusion logic here.
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
# DEPTH VISUALIZATION
# ============================================================

def make_depth_vis(depth_map):

    d = depth_map.copy()

    d = (
        (d - d.min()) /
        (d.max() - d.min() + 1e-6)
    )

    d = (
        d * 255
    ).astype(np.uint8)

    depth_vis = cv2.applyColorMap(
        d,
        cv2.COLORMAP_MAGMA
    )

    return depth_vis


# ============================================================
# MAIN
# ============================================================

def run():

    print("=" * 70)
    print("Single KITTI Image Analysis")
    print("=" * 70)

    image_path = os.path.join(
        KITTI_ROOT,
        "data_object_image_2",
        "training",
        "image_2",
        IMAGE_ID + ".png"
    )

    label_path = os.path.join(
        KITTI_ROOT,
        "data_object_label_2",
        "training",
        "label_2",
        IMAGE_ID + ".txt"
    )

    print("Loading image from:", image_path)
    print("Loading label from:", label_path)

    if not os.path.exists(image_path):

        print("[ERROR] Image path does not exist.")
        return

    if not os.path.exists(label_path):

        print("[ERROR] Label path does not exist.")
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

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        print("[ERROR] Image could not be read.")
        return

    # --------------------------------------------------------
    # Parse GT labels
    # --------------------------------------------------------

    gt_labels = [
        label
        for label in parse_kitti_label_file(label_path)
        if label.is_valid and label.adas_class
    ]

    print(f"Valid GT objects: {len(gt_labels)}")

    # --------------------------------------------------------
    # Run detection + depth
    # --------------------------------------------------------

    det_output = detector.process(
        frame
    )

    depth_out = depth_estimator.estimate(
        frame
    )

    print(f"Detected ADAS objects: {len(det_output.detections)}")
    print(f"Depth inference: {depth_out.processing_ms:.1f} ms")

    vis = frame.copy()

    matched_pred = set()

    rows = []

    print("\nPer-object comparison:")
    print("-" * 90)

    print(
        f"{'Class':<10} "
        f"{'GT(m)':>8} "
        f"{'Geom(m)':>10} "
        f"{'Fusion(m)':>11} "
        f"{'G-Err':>8} "
        f"{'F-Err':>8} "
        f"{'IoU':>6}"
    )

    print("-" * 90)

    # --------------------------------------------------------
    # Match GT objects to detections
    # --------------------------------------------------------

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

        if best_det_idx is None:
            continue

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

        fused_dist = fusion_distance(
            det,
            depth_out
        )

        geom_err = abs(
            geom_dist - gt_dist
        )

        fusion_err = abs(
            fused_dist - gt_dist
        )

        rows.append(
            {
                "class": det.class_name,
                "gt": gt_dist,
                "geometry": geom_dist,
                "fusion": fused_dist,
                "geom_err": geom_err,
                "fusion_err": fusion_err,
                "iou": best_iou
            }
        )

        # ----------------------------------------------------
        # Draw detection box
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            det.bbox
        )

        color = (
            (0, 255, 0)
            if fusion_err < geom_err
            else (0, 0, 255)
        )

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = (
            f"{det.class_name} "
            f"GT:{gt_dist:.1f} "
            f"G:{geom_dist:.1f} "
            f"F:{fused_dist:.1f}"
        )

        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA
        )

        print(
            f"{det.class_name:<10} "
            f"{gt_dist:>8.1f} "
            f"{geom_dist:>10.1f} "
            f"{fused_dist:>11.1f} "
            f"{geom_err:>8.1f} "
            f"{fusion_err:>8.1f} "
            f"{best_iou:>6.2f}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("-" * 90)

    if len(rows) == 0:

        print(
            "[WARNING] No GT-detection matches found. "
            "Try lowering IOU_THRESHOLD to 0.3 or use another IMAGE_ID."
        )

    else:

        geom_mae = np.mean(
            [r["geom_err"] for r in rows]
        )

        fusion_mae = np.mean(
            [r["fusion_err"] for r in rows]
        )

        improvement = (
            (geom_mae - fusion_mae) /
            (geom_mae + 1e-6)
        ) * 100

        print(f"Matched objects: {len(rows)}")
        print(f"Geometry MAE: {geom_mae:.2f} m")
        print(f"Fusion MAE:   {fusion_mae:.2f} m")
        print(f"Improvement:  {improvement:.2f}%")

    # --------------------------------------------------------
    # Depth map
    # --------------------------------------------------------

    depth_vis = make_depth_vis(
        depth_out.depth_map
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 7)
    )

    axes[0].imshow(
        cv2.cvtColor(
            vis,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[0].set_title(
        "Detection + GT vs Geometry vs Fusion"
    )

    axes[0].axis(
        "off"
    )

    axes[1].imshow(
        cv2.cvtColor(
            depth_vis,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[1].set_title(
        "Depth Anything Map"
    )

    axes[1].axis(
        "off"
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{IMAGE_ID}_analysis.png"
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print("\nSaved:", save_path)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    run()

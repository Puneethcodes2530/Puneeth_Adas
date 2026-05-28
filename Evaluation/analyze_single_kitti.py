"""
NeuroSentinel v3
Single KITTI Image Analysis

Compares:
1. Ground Truth Distance
2. Geometry-only Distance
3. Depth Fusion Distance

for ONE image.

Outputs:
- visual comparison
- per-object errors
- depth map
"""

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
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

from perception.adaptive_detector import AdaptiveDetector
from perception.depth_estimator_da import DepthEstimatorDA


# ============================================================
# CONFIG
# ============================================================

KITTI_ROOT = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI"

# CHANGE THIS IMAGE ID
IMAGE_ID = "007069"

IOU_THRESHOLD = 0.3


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
# LABEL PARSER
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

    # ========================================================
    # BASE GEOMETRY DISTANCE
    # ========================================================

    geom_dist = geometry_distance(det)

    # ========================================================
    # BBOX
    # ========================================================

    x1, y1, x2, y2 = map(int, det.bbox)

    h, w = depth_out.depth_map.shape

    # ========================================================
    # SAFETY CLAMP
    # ========================================================

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))

    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    # Invalid box
    if x2 <= x1 or y2 <= y1:
        return geom_dist

    # ========================================================
    # FOOT ROI
    # ========================================================
    #
    # Bottom region is more stable
    # for road-contact objects.
    # ========================================================

    foot_y = y2

    y_start = max(
        foot_y - 12,
        0
    )

    roi = depth_out.depth_map[
        y_start:foot_y,
        x1:x2
    ]

    # Empty ROI
    if roi.size == 0:
        return geom_dist

    # ========================================================
    # DEPTH STATISTICS
    # ========================================================

    depth_mean = float(
        np.mean(roi)
    )

    depth_std = float(
        np.std(roi)
    )

    # ========================================================
    # DEPTH STABILITY
    # ========================================================
    #
    # Lower std = more stable
    # ========================================================

    stability = 1.0 - min(
        1.0,
        depth_std / (depth_mean + 1e-6)
    )

    # ========================================================
    # ROBUST DEPTH VALUE
    # ========================================================

    raw_depth = float(
        np.percentile(roi, 20)
    )

    raw_depth = max(raw_depth, 0.05)

    # ========================================================
    # SMALL RELATIVE CORRECTION
    # ========================================================
    #
    # DO NOT aggressively invert depth.
    #
    # Depth Anything gives:
    # relative depth
    #
    # not exact metric metres.
    #
    # So:
    # use depth as slight refinement only.
    # ========================================================

    correction = 1.0 + (
        (0.5 - raw_depth) * 0.25
    )

    correction = np.clip(
        correction,
        0.85,
        1.15
    )

    # ========================================================
    # DEPTH-REFINED DISTANCE
    # ========================================================

    depth_adjusted = (
        geom_dist * correction
    )

    # ========================================================
    # CONFIDENCE-WEIGHTED FUSION
    # ========================================================
    #
    # If depth unstable:
    # trust geometry more
    # ========================================================

    fusion_weight = 0.15 * stability

    final_dist = (

        (1 - fusion_weight) * geom_dist +

        fusion_weight * depth_adjusted
    )

    # ========================================================
    # FINAL SAFETY CLAMP
    # ========================================================

    final_dist = np.clip(
        final_dist,
        1.0,
        120.0
    )

    return float(final_dist)


# ============================================================
# MAIN
# ============================================================

def run():

    print("=" * 60)
    print("Single KITTI Image Analysis")
    print("=" * 60)

    image_path = os.path.join(

        KITTI_ROOT,

        "data_object_image_2",

        "training",

        "image_2",

        IMAGE_ID + ".png"
    )

    
    print("Loading from:", image_path)  # ✅ ADD THIS LINE


    label_path = os.path.join(

        KITTI_ROOT,

        "data_object_label_2",

        "training",

        "label_2",

        IMAGE_ID + ".txt"
    )

    print("\nLoading models...")

    detector = AdaptiveDetector(
        'yolov8s.pt'
    )

    depth_estimator = DepthEstimatorDA()

    print("✓ Models loaded")

    frame = cv2.imread(image_path)

    if frame is None:

        print("Image not found")
        return

    gt_labels = [

        l for l in
        parse_kitti_label_file(label_path)

        if l.is_valid and l.adas_class
    ]

    det_output = detector.process(frame)

    depth_out = depth_estimator.estimate(frame)

    vis = frame.copy()

    print("\nPer-object comparison:")
    print("-" * 60)

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

        if best_det is None:
            continue

        matched_pred.add(best_det)

        det = det_output.detections[
            best_det
        ]

        gt_dist = gt.z_3d

        geom_dist = geometry_distance(det)

        fusion_dist = fusion_distance(
            det,
            depth_out
        )

        geom_err = abs(
            geom_dist - gt_dist
        )

        fusion_err = abs(
            fusion_dist - gt_dist
        )

        # ----------------------------------------------------
        # DRAW
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            det.bbox
        )

        if fusion_err < geom_err:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

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

            f"F:{fusion_dist:.1f}"
        )

        cv2.putText(

            vis,

            label,

            (x1, max(20, y1 - 10)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            color,

            2
        )

        print(

            f"{det.class_name:<10}"

            f"GT:{gt_dist:>6.1f}m "

            f"G:{geom_dist:>6.1f}m "

            f"F:{fusion_dist:>6.1f}m "

            f"G-Err:{geom_err:>5.1f} "

            f"F-Err:{fusion_err:>5.1f}"
        )

    # ========================================================
    # DEPTH MAP
    # ========================================================

    depth_vis = (
        depth_out.depth_map * 255
    ).astype(np.uint8)

    depth_vis = cv2.applyColorMap(
        depth_vis,
        cv2.COLORMAP_MAGMA
    )

    # ========================================================
    # PLOT
    # ========================================================

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
        "Detection + Comparison"
    )

    axes[0].axis('off')

    axes[1].imshow(
        cv2.cvtColor(
            depth_vis,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[1].set_title(
        "Depth Map"
    )

    axes[1].axis('off')

    plt.tight_layout()

    os.makedirs(
        "outputs/single_analysis",
        exist_ok=True
    )

    save_path = os.path.join(

        "outputs/single_analysis",

        f"{IMAGE_ID}_analysis.png"
    )

    plt.savefig(
        save_path,
        dpi=150
    )

    plt.show()

    print(f"\nSaved: {save_path}")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    run()
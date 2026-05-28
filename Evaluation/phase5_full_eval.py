import cv2
import numpy as np
import glob
import os
import sys
import json
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

sys.path.insert(0, os.path.abspath(
    os.path.dirname(os.path.dirname(__file__))
))

from ultralytics import YOLO

from perception.tracker import (
    Phase4TrackerPipeline
)

from perception.depth_estimator import (
    DepthEstimatorDA
)

from perception.scene_analyzer import (
    CLIPSceneDetector
)

# =========================================================
# PATHS
# =========================================================

BDD_PATH = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test"

KITTI_IMG = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI\data_object_image_2\training\image_2"

KITTI_LBL = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI\data_object_label_2\training\label_2"

N_IMAGES = 100

# =========================================================
# DEPTH ENGINE
# =========================================================

depth_engine = DepthEstimatorDA()

def my_depth_fn(frame):

    depth_output = depth_engine.estimate(frame)

    return depth_output.depth_map

# =========================================================
# LOAD PIPELINE
# =========================================================

pipe = Phase4TrackerPipeline(
    depth_fn=my_depth_fn
)

# =========================================================
# IOU
# =========================================================

def iou(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])

    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    if inter <= 0:
        return 0.0

    areaA = (a[2]-a[0]) * (a[3]-a[1])

    areaB = (b[2]-b[0]) * (b[3]-b[1])

    return inter / (areaA + areaB - inter + 1e-6)

# =========================================================
# KITTI LABEL PARSER
# =========================================================

def parse_kitti_label(path):

    CLASS_MAP = {

        'Car': 'car',
        'Van': 'car',
        'Truck': 'truck',
        'Pedestrian': 'person',
        'Cyclist': 'bicycle'
    }

    objs = []

    if not os.path.exists(path):
        return objs

    with open(path) as f:

        for line in f:

            p = line.strip().split()

            if len(p) < 15:
                continue

            cls = CLASS_MAP.get(p[0])

            if cls is None:
                continue

            objs.append({

                'class': cls,

                'bbox': [

                    float(p[4]),
                    float(p[5]),
                    float(p[6]),
                    float(p[7])
                ],

                'distance_m': float(p[13])
            })

    return objs

# =========================================================
# DETECTION EVALUATION
# =========================================================

def run_detection_eval():

    print("\nRunning detection evaluation...")

    model = YOLO('yolov8x.pt')

    img_files = sorted(
        glob.glob(f"{KITTI_IMG}/*.png")
    )[:N_IMAGES]

    stats = defaultdict(
        lambda: {'tp':0,'fp':0,'fn':0}
    )

    for idx, img_path in enumerate(img_files):

        frame = cv2.imread(img_path)

        if frame is None:
            continue

        name = os.path.splitext(
            os.path.basename(img_path)
        )[0]

        gt_path = os.path.join(
            KITTI_LBL,
            name + ".txt"
        )

        gt_objs = parse_kitti_label(gt_path)

        results = model(

            frame,

            conf=0.30,

            verbose=False
        )

        preds = results[0].boxes

        matched_pred = set()

        for gt in gt_objs:

            best_iou = 0.5

            best_idx = None

            for p_i in range(len(preds)):

                if p_i in matched_pred:
                    continue

                cls = model.names[
                    int(preds.cls[p_i])
                ]

                if cls != gt['class']:
                    continue

                pbox = preds.xyxy[p_i].tolist()

                score = iou(
                    gt['bbox'],
                    pbox
                )

                if score > best_iou:

                    best_iou = score

                    best_idx = p_i

            if best_idx is not None:

                matched_pred.add(best_idx)

                stats[gt['class']]['tp'] += 1

            else:

                stats[gt['class']]['fn'] += 1

        for p_i in range(len(preds)):

            if p_i not in matched_pred:

                cls = model.names[
                    int(preds.cls[p_i])
                ]

                if cls in stats:

                    stats[cls]['fp'] += 1

    summary = {}

    print("\n================================================")
    print("DETECTION METRICS")
    print("================================================")

    for cls, s in stats.items():

        tp = s['tp']
        fp = s['fp']
        fn = s['fn']

        precision = tp / (tp+fp+1e-6)

        recall = tp / (tp+fn+1e-6)

        f1 = 2 * precision * recall / (
            precision + recall + 1e-6
        )

        summary[cls] = {

            'precision': round(precision,3),

            'recall': round(recall,3),

            'f1': round(f1,3)
        }

        print(

            f"{cls:<12}"

            f"P:{precision:.3f} "

            f"R:{recall:.3f} "

            f"F1:{f1:.3f}"
        )

    return summary

# =========================================================
# LATENCY
# =========================================================

def run_latency():

    print("\nRunning latency benchmark...")

    imgs = sorted(
        glob.glob(f"{BDD_PATH}/*.jpg")
    )[:50]

    latencies = []

    for idx, img_path in enumerate(imgs):

        frame = cv2.imread(img_path)

        if frame is None:
            continue

        t = time.perf_counter()

        pipe.process(frame)

        ms = (
            time.perf_counter()-t
        ) * 1000

        latencies.append(ms)

        print(

            f"[{idx+1}/{len(imgs)}] "

            f"{ms:.0f}ms"
        )

    arr = np.array(latencies)

    result = {

        'p50': round(np.percentile(arr,50),1),

        'p90': round(np.percentile(arr,90),1),

        'p99': round(np.percentile(arr,99),1),

        'mean': round(np.mean(arr),1),

        'fps': round(1000/np.mean(arr),2)
    }

    print("\n================================================")
    print("LATENCY METRICS")
    print("================================================")

    for k,v in result.items():

        print(f"{k:<10}: {v}")

    return result

# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    det = run_detection_eval()

    lat = run_latency()

    # =====================================================
    # PLOT
    # =====================================================

    fig = plt.figure(figsize=(16,8))

    fig.patch.set_facecolor('#0a0a1a')

    gs = gridspec.GridSpec(1,2)

    ax1 = fig.add_subplot(gs[0,0])

    ax2 = fig.add_subplot(gs[0,1])

    # -----------------------------------------------------

    classes = list(det.keys())

    precs = [
        det[c]['precision']
        for c in classes
    ]

    recs = [
        det[c]['recall']
        for c in classes
    ]

    f1s = [
        det[c]['f1']
        for c in classes
    ]

    x = np.arange(len(classes))

    w = 0.25

    ax1.bar(x-w, precs, w)

    ax1.bar(x, recs, w)

    ax1.bar(x+w, f1s, w)

    ax1.set_xticks(x)

    ax1.set_xticklabels(classes)

    ax1.set_ylim(0,1.1)

    ax1.set_title(
        "Detection Metrics"
    )

    # -----------------------------------------------------

    lat_keys = list(lat.keys())

    lat_vals = list(lat.values())

    ax2.bar(lat_keys, lat_vals)

    ax2.set_title(
        "Latency Metrics"
    )

    # -----------------------------------------------------

    plt.tight_layout()

    plt.savefig(

        "outputs/reports/phase5_report.png",

        dpi=180
    )

    plt.show()

    # =====================================================
    # SAVE JSON
    # =====================================================

    final = {

        'detection': det,

        'latency': lat
    }

    with open(

        "outputs/reports/phase5_report.json",

        'w'

    ) as f:

        json.dump(final, f, indent=2)

    print("\n================================================")
    print("PHASE 5 COMPLETE")
    print("================================================")

    print("\nSaved:")

    print(
        "outputs/reports/phase5_report.png"
    )

    print(
        "outputs/reports/phase5_report.json"
    )

# =========================================================

if __name__ == "__main__":

    main()
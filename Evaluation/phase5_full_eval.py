"""
NeuroSentinel v3 — Phase 5 Full Evaluation

Evaluates:
1. KITTI object detection metrics
2. End-to-end pipeline latency
3. Visual sample outputs with tracking, distance, TTC and risk

Outputs:
- phase5_report.png
- phase5_report.json
- visual_samples/*.png
"""

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


# =========================================================
# PROJECT ROOT
# =========================================================

ROOT = os.path.abspath(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

sys.path.insert(0, ROOT)

print("Project root:", ROOT)


# =========================================================
# IMPORTS
# =========================================================

from ultralytics import YOLO

from perception.tracker import Phase4TrackerPipeline
from perception.depth_estimator import DepthEstimatorDA



# =========================================================
# PATHS
# =========================================================

BDD_PATH = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test"

KITTI_IMG = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI\data_object_image_2\training\image_2"

KITTI_LBL = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI\data_object_label_2\training\label_2"

N_IMAGES = 100
N_LATENCY_IMAGES = 10
N_VISUAL_SAMPLES = 5

OUTPUT_DIR = os.path.join(ROOT, "outputs", "reports")
VIS_DIR = os.path.join(OUTPUT_DIR, "visual_samples")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


# =========================================================
# DEPTH ENGINE
# =========================================================

print("[INFO] Loading Depth Engine...")
depth_engine = DepthEstimatorDA()


def my_depth_fn(frame):
    """
    Function passed into Phase4TrackerPipeline.
    It returns only the depth map.
    """
    depth_output = depth_engine.estimate(frame)
    return depth_output.depth_map


# =========================================================
# LOAD PHASE 4 PIPELINE
# =========================================================

print("[INFO] Loading Phase 4 pipeline...")

pipe = Phase4TrackerPipeline(
    depth_fn=my_depth_fn
)

print("[INFO] Phase 4 pipeline loaded.")


# =========================================================
# IOU FUNCTION
# =========================================================

def iou(a, b):
    """
    Computes Intersection over Union between two boxes.

    a = [x1, y1, x2, y2]
    b = [x1, y1, x2, y2]
    """

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])

    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    if inter <= 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    union = area_a + area_b - inter

    return inter / (union + 1e-6)


# =========================================================
# KITTI LABEL PARSER
# =========================================================

def parse_kitti_label(path):
    """
    Parses KITTI object label txt file.

    Extracts:
    - class
    - 2D bbox
    - z distance in meters
    """

    class_map = {
        "Car": "car",
        "Van": "car",
        "Truck": "truck",
        "Pedestrian": "person",
        "Person_sitting": "person",
        "Cyclist": "bicycle",
        "Tram": "bus"
    }

    objs = []

    if not os.path.exists(path):
        return objs

    with open(path, "r") as f:

        for line in f:

            p = line.strip().split()

            if len(p) < 15:
                continue

            if p[0] == "DontCare":
                continue

            cls = class_map.get(p[0])

            if cls is None:
                continue

            truncated = float(p[1])
            occluded = int(p[2])
            z_3d = float(p[13])

            # Same fair filtering logic
            if truncated >= 0.5:
                continue

            if occluded >= 3:
                continue

            if z_3d <= 0 or z_3d >= 80:
                continue

            objs.append({
                "class": cls,
                "bbox": [
                    float(p[4]),
                    float(p[5]),
                    float(p[6]),
                    float(p[7])
                ],
                "distance_m": z_3d
            })

    return objs


# =========================================================
# DETECTION EVALUATION
# =========================================================

def run_detection_eval():

    print("\n================================================")
    print("Running KITTI Detection Evaluation")
    print("================================================")

    model = YOLO("yolov8x.pt")

    img_files = sorted(
        glob.glob(os.path.join(KITTI_IMG, "*.png"))
    )[:N_IMAGES]

    if len(img_files) == 0:
        print("[ERROR] No KITTI images found.")
        print("Check KITTI_IMG path:", KITTI_IMG)
        return {}

    stats = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
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

        # ---------------------------------------------
        # Match GT objects with predictions
        # ---------------------------------------------
        for gt in gt_objs:

            best_iou = 0.5
            best_idx = None

            for p_i in range(len(preds)):

                if p_i in matched_pred:
                    continue

                pred_cls = model.names[
                    int(preds.cls[p_i])
                ]

                if pred_cls != gt["class"]:
                    continue

                pred_box = preds.xyxy[p_i].tolist()

                score = iou(
                    gt["bbox"],
                    pred_box
                )

                if score > best_iou:
                    best_iou = score
                    best_idx = p_i

            if best_idx is not None:

                matched_pred.add(best_idx)
                stats[gt["class"]]["tp"] += 1

            else:

                stats[gt["class"]]["fn"] += 1

        # ---------------------------------------------
        # Count false positives
        # ---------------------------------------------
        for p_i in range(len(preds)):

            if p_i in matched_pred:
                continue

            pred_cls = model.names[
                int(preds.cls[p_i])
            ]

            if pred_cls in ["person", "bicycle", "car", "motorcycle", "bus", "truck"]:
                stats[pred_cls]["fp"] += 1

        print(f"[{idx + 1}/{len(img_files)}] {name}")

    # ---------------------------------------------
    # Compute metrics
    # ---------------------------------------------
    summary = {}

    print("\n================================================")
    print("DETECTION METRICS")
    print("================================================")

    for cls, s in stats.items():

        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        f1 = (
            2 * precision * recall /
            (precision + recall + 1e-6)
        )

        summary[cls] = {
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f1": round(float(f1), 3),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn)
        }

        print(
            f"{cls:<12}"
            f"P:{precision:.3f} "
            f"R:{recall:.3f} "
            f"F1:{f1:.3f} "
            f"TP:{tp} FP:{fp} FN:{fn}"
        )

    return summary


# =========================================================
# LATENCY BENCHMARK
# =========================================================

def run_latency():

    print("\n================================================")
    print("Running End-to-End Latency Benchmark")
    print("================================================")

    imgs = sorted(
        glob.glob(os.path.join(BDD_PATH, "*.jpg"))
    )[:N_LATENCY_IMAGES]

    if len(imgs) == 0:
        print("[WARNING] No BDD images found. Trying PNG fallback...")

        imgs = sorted(
            glob.glob(os.path.join(BDD_PATH, "*.png"))
        )[:N_LATENCY_IMAGES]

    if len(imgs) == 0:
        print("[ERROR] No images found for latency benchmark.")
        print("Check BDD_PATH:", BDD_PATH)

        return {
            "p50": 0,
            "p90": 0,
            "p99": 0,
            "mean": 0,
            "fps": 0
        }

    latencies = []

    for idx, img_path in enumerate(imgs):

        frame = cv2.imread(img_path)

        if frame is None:
            continue

        t = time.perf_counter()

        objects, depth_map, latency = pipe.process(frame)

        ms = (time.perf_counter() - t) * 1000

        latencies.append(ms)

        print(
            f"[{idx + 1}/{len(imgs)}] "
            f"{os.path.basename(img_path)} "
            f"{ms:.0f}ms | objects={len(objects)}"
        )

    arr = np.array(latencies)

    if len(arr) == 0:
        return {
            "p50": 0,
            "p90": 0,
            "p99": 0,
            "mean": 0,
            "fps": 0
        }

    result = {
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p90_ms": round(float(np.percentile(arr, 90)), 1),
        "p99_ms": round(float(np.percentile(arr, 99)), 1),
        "mean_ms": round(float(np.mean(arr)), 1),
        "fps": round(float(1000 / np.mean(arr)), 2)
    }

    print("\n================================================")
    print("LATENCY METRICS")
    print("================================================")

    for k, v in result.items():
        print(f"{k:<12}: {v}")

    return result


# =========================================================
# VISUAL SAMPLE GENERATION
# =========================================================

def visualize_samples():

    print("\n================================================")
    print("Generating Visual Samples")
    print("================================================")

    img_files = sorted(
        glob.glob(os.path.join(BDD_PATH, "*.jpg"))
    )[:N_VISUAL_SAMPLES]

    if len(img_files) == 0:

        img_files = sorted(
            glob.glob(os.path.join(KITTI_IMG, "*.png"))
        )[:N_VISUAL_SAMPLES]

    if len(img_files) == 0:
        print("[ERROR] No images found for visualization.")
        return []

    saved_paths = []

    for idx, img_path in enumerate(img_files):

        frame = cv2.imread(img_path)

        if frame is None:
            continue

        objects, depth_map, latency = pipe.process(frame)

        vis = pipe.draw(
            frame,
            objects,
            latency
        )

        # ---------------------------------------------
        # Depth visualization
        # ---------------------------------------------
        if depth_map is not None:

            d = depth_map.copy()

            d = (
                (d - d.min()) /
                (d.max() - d.min() + 1e-6)
            )

            d = (d * 255).astype(np.uint8)

            depth_color = cv2.applyColorMap(
                d,
                cv2.COLORMAP_MAGMA
            )

            if depth_color.shape[:2] != vis.shape[:2]:
                depth_color = cv2.resize(
                    depth_color,
                    (vis.shape[1], vis.shape[0])
                )

            combined = np.hstack([
                vis,
                depth_color
            ])

        else:

            combined = vis

        save_path = os.path.join(
            VIS_DIR,
            f"sample_{idx + 1}_{os.path.basename(img_path)}"
        )

        cv2.imwrite(save_path, combined)

        saved_paths.append(save_path)

        print(
            f"Saved visual sample {idx + 1}: "
            f"{save_path}"
        )

    return saved_paths


# =========================================================
# PLOT REPORT
# =========================================================

def plot_report(det, lat):

    print("\n================================================")
    print("Generating Report Plot")
    print("================================================")

    fig = plt.figure(
        figsize=(18, 8),
        facecolor="#0a0a1a"
    )

    gs = gridspec.GridSpec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.set_facecolor("#111122")
    ax2.set_facecolor("#111122")

    # ---------------------------------------------
    # Detection chart
    # ---------------------------------------------
    classes = list(det.keys())

    if len(classes) == 0:
        classes = ["none"]
        precs = [0]
        recs = [0]
        f1s = [0]
    else:
        precs = [det[c]["precision"] for c in classes]
        recs = [det[c]["recall"] for c in classes]
        f1s = [det[c]["f1"] for c in classes]

    x = np.arange(len(classes))
    w = 0.25

    b1 = ax1.bar(
        x - w,
        precs,
        w,
        label="Precision",
        color="#00BFFF"
    )

    b2 = ax1.bar(
        x,
        recs,
        w,
        label="Recall",
        color="#FFA500"
    )

    b3 = ax1.bar(
        x + w,
        f1s,
        w,
        label="F1",
        color="#32CD32"
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        classes,
        color="white",
        rotation=20
    )

    ax1.set_ylim(0, 1.1)

    ax1.set_title(
        "KITTI Detection Metrics",
        color="white",
        fontsize=14,
        fontweight="bold"
    )

    ax1.set_ylabel(
        "Score",
        color="white"
    )

    ax1.tick_params(
        axis="y",
        colors="white"
    )

    ax1.grid(
        axis="y",
        alpha=0.25
    )

    ax1.legend(
        facecolor="#111122",
        edgecolor="white",
        labelcolor="white"
    )

    ax1.bar_label(
        b1,
        fmt="%.2f",
        color="white",
        fontsize=8
    )

    ax1.bar_label(
        b2,
        fmt="%.2f",
        color="white",
        fontsize=8
    )

    ax1.bar_label(
        b3,
        fmt="%.2f",
        color="white",
        fontsize=8
    )

    # ---------------------------------------------
    # Latency chart
    # ---------------------------------------------
    latency_display = {
        "p50": lat.get("p50_ms", 0),
        "p90": lat.get("p90_ms", 0),
        "p99": lat.get("p99_ms", 0),
        "mean": lat.get("mean_ms", 0)
    }

    lat_keys = list(latency_display.keys())
    lat_vals = list(latency_display.values())

    bars = ax2.bar(
        lat_keys,
        lat_vals,
        color="#FF4C4C"
    )

    ax2.set_title(
        "End-to-End Latency Metrics",
        color="white",
        fontsize=14,
        fontweight="bold"
    )

    ax2.set_ylabel(
        "Latency (ms)",
        color="white"
    )

    ax2.tick_params(
        axis="x",
        colors="white"
    )

    ax2.tick_params(
        axis="y",
        colors="white"
    )

    ax2.grid(
        axis="y",
        alpha=0.25
    )

    ax2.bar_label(
        bars,
        fmt="%.0f",
        color="white",
        fontsize=9
    )

    fps_text = f"FPS: {lat.get('fps', 0):.2f}"

    ax2.text(
        0.5,
        0.92,
        fps_text,
        transform=ax2.transAxes,
        ha="center",
        color="yellow",
        fontsize=14,
        fontweight="bold"
    )

    fig.suptitle(
        "NeuroSentinel v3 — Phase 5 Evaluation Report",
        color="white",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "phase5_report.png"
    )

    plt.savefig(
        save_path,
        dpi=180,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )

    plt.show()

    print("Saved plot:", save_path)

    return save_path


# =========================================================
# SAVE JSON REPORT
# =========================================================

def save_json_report(det, lat, visual_paths):

    final = {
        "detection": det,
        "latency": lat,
        "visual_samples": visual_paths,
        "notes": {
            "detection_eval": "KITTI object dataset using IoU > 0.5 matching",
            "latency_eval": "End-to-end Phase4 pipeline latency on sample BDD/KITTI images",
            "visualization": "Annotated images include tracking ID, class, confidence, distance, TTC and risk level"
        }
    }

    save_path = os.path.join(
        OUTPUT_DIR,
        "phase5_report.json"
    )

    with open(save_path, "w") as f:
        json.dump(final, f, indent=2)

    print("Saved JSON:", save_path)

    return save_path


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n================================================")
    print("PHASE 5 FULL EVALUATION STARTED")
    print("================================================")

    det = run_detection_eval()

    lat = run_latency()

    visual_paths = visualize_samples()

    plot_path = plot_report(
        det,
        lat
    )

    json_path = save_json_report(
        det,
        lat,
        visual_paths
    )

    print("\n================================================")
    print("PHASE 5 COMPLETE")
    print("================================================")

    print("\nSaved outputs:")
    print(plot_path)
    print(json_path)

    for p in visual_paths:
        print(p)


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()

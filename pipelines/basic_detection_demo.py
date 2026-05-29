"""
NeuroSentinel v3 — Basic Detection Demo

Purpose:
- Run baseline YOLO detection on sample images
- Select images from configured datasets
- Draw ADAS-relevant detections
- Save visualization grid
- Save individual frames
- Save JSON summary

Outputs:
- outputs/day3_first_detection.png
- outputs/day3_detection_output.json
- outputs/frames/*.png
"""

import os
import sys
import glob
import json
import time
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO


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
# CONFIG
# ============================================================

CONFIG_PATH = os.path.join(
    ROOT,
    "configs",
    "paths.yaml"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs"
)

FRAME_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "frames"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    FRAME_OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_WEIGHTS = "yolov8s.pt"      # faster and safer
# MODEL_WEIGHTS = "yolov8x.pt"    # higher accuracy, slower

CONF_THRESHOLD = 0.35
N_IMAGES = 6
BUDGET_MS = 40.0


# ============================================================
# ADAS CLASSES
# ============================================================

ADAS_CLASSES = {
    "person": (255, 50, 50),
    "bicycle": (255, 165, 0),
    "motorcycle": (255, 140, 0),
    "car": (50, 205, 50),
    "truck": (0, 128, 255),
    "bus": (128, 0, 255),
    "traffic light": (255, 255, 0),
    "stop sign": (255, 0, 128),
}


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config():

    if not os.path.exists(CONFIG_PATH):

        raise FileNotFoundError(
            f"Config file not found: {CONFIG_PATH}"
        )

    with open(
        CONFIG_PATH,
        "r"
    ) as f:

        config = yaml.safe_load(f)

    if config is None:

        raise ValueError(
            "configs/paths.yaml is empty or invalid"
        )

    return config


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(
    folder,
    limit=6
):
    """
    Recursively find images inside folder.
    Supports jpg, jpeg, png.
    """

    if folder is None:

        return []

    if not os.path.exists(folder):

        print(
            f"[WARNING] Folder does not exist: {folder}"
        )

        return []

    patterns = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.png"
    ]

    images = []

    for pattern in patterns:

        images.extend(
            glob.glob(
                os.path.join(folder, pattern),
                recursive=True
            )
        )

    images = sorted(
        list(set(images))
    )

    return images[:limit]


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    img_bgr,
    result,
    model
):
    """
    Draws ADAS-relevant detections on image.

    Returns:
    - annotated RGB image
    - detections list
    - class_counts dict
    """

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    class_counts = {}
    detections = []

    if result.boxes is None:

        return img_rgb, detections, class_counts

    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        conf = float(
            box.conf[0].item()
        )

        cls_id = int(
            box.cls[0].item()
        )

        cls_name = model.names[
            cls_id
        ]

        if cls_name not in ADAS_CLASSES:

            continue

        color = ADAS_CLASSES[
            cls_name
        ]

        # ----------------------------------------------------
        # Draw box
        # ----------------------------------------------------
        cv2.rectangle(
            img_rgb,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = (
            f"{cls_name} {conf:.2f}"
        )

        label_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1
        )[0]

        label_y = max(
            y1,
            label_size[1] + 8
        )

        cv2.rectangle(
            img_rgb,
            (x1, label_y - label_size[1] - 8),
            (x1 + label_size[0] + 4, label_y),
            color,
            -1
        )

        cv2.putText(
            img_rgb,
            label,
            (x1 + 2, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        class_counts[cls_name] = (
            class_counts.get(cls_name, 0) + 1
        )

        detections.append(
            {
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "bbox_width_px": int(x2 - x1),
                "bbox_height_px": int(y2 - y1)
            }
        )

    return img_rgb, detections, class_counts


# ============================================================
# SELECT DATASET IMAGES
# ============================================================

def select_images_from_config(config):

    datasets = config.get(
        "datasets",
        {}
    )

    image_sources = []

    # --------------------------------------------------------
    # Add sources safely only if they exist in YAML
    # --------------------------------------------------------
    if "kitti" in datasets:

        image_sources.append(
            (
                "KITTI",
                datasets["kitti"].get("training_images")
            )
        )

    if "bdd100k" in datasets:

        image_sources.append(
            (
                "BDD100K",
                datasets["bdd100k"].get("images_testA")
            )
        )

    if "ua_detrac" in datasets:

        image_sources.append(
            (
                "UA-DETRAC",
                datasets["ua_detrac"].get("train_images")
            )
        )

    if "idd" in datasets:

        image_sources.append(
            (
                "IDD",
                datasets["idd"].get("images")
            )
        )

    # --------------------------------------------------------
    # Find first available dataset
    # --------------------------------------------------------
    for source_name, source_path in image_sources:

        imgs = find_images(
            source_path,
            limit=N_IMAGES
        )

        if len(imgs) > 0:

            print(
                f"\n✓ Using {source_name}: "
                f"found {len(imgs)} images"
            )

            return source_name, imgs

    return None, []


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NeuroSentinel v3 — Basic Detection Demo")
    print("=" * 70)

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------
    config = load_config()

    selected_source, selected_images = select_images_from_config(
        config
    )

    if len(selected_images) == 0:

        print(
            "\n[ERROR] No images found. "
            "Check paths in configs/paths.yaml"
        )

        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print(
        f"\nLoading YOLO model: {MODEL_WEIGHTS}"
    )

    model = YOLO(
        MODEL_WEIGHTS
    )

    print("✓ Model loaded")
    print(
        f"Classes preview: "
        f"{list(model.names.values())[:10]}"
    )

    print(
        f"\nRunning detection on "
        f"{len(selected_images)} images..."
    )

    results_summary = []
    latencies = []

    # --------------------------------------------------------
    # Plot setup
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10)
    )

    fig.suptitle(
        f"NeuroSentinel v3 — First Detection\n"
        f"Dataset: {selected_source} | Model: {MODEL_WEIGHTS}",
        fontsize=14,
        fontweight="bold"
    )

    axes = axes.flatten()

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------
    for idx, img_path in enumerate(
        selected_images[:N_IMAGES]
    ):

        img_bgr = cv2.imread(
            img_path
        )

        if img_bgr is None:

            print(
                f"[WARNING] Could not read: {img_path}"
            )

            continue

        # ----------------------------------------------------
        # Inference timing
        # ----------------------------------------------------
        t_start = time.perf_counter()

        results = model(
            img_bgr,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        latency_ms = (
            time.perf_counter() -
            t_start
        ) * 1000

        latencies.append(
            latency_ms
        )

        result = results[0]

        img_rgb, detections, class_counts = draw_detections(
            img_bgr,
            result,
            model
        )

        # ----------------------------------------------------
        # Save individual frame
        # ----------------------------------------------------
        frame_save_path = os.path.join(
            FRAME_OUTPUT_DIR,
            f"basic_{idx + 1}_{os.path.basename(img_path)}"
        )

        cv2.imwrite(
            frame_save_path,
            cv2.cvtColor(
                img_rgb,
                cv2.COLOR_RGB2BGR
            )
        )

        # ----------------------------------------------------
        # Add subplot
        # ----------------------------------------------------
        if idx < len(axes):

            axes[idx].imshow(
                img_rgb
            )

            title_parts = [
                f"{cls}: {cnt}"
                for cls, cnt in class_counts.items()
            ]

            title = (
                f"{os.path.basename(img_path)[:22]}\n"
                f"{', '.join(title_parts) if title_parts else 'No ADAS objects'}\n"
                f"Latency: {latency_ms:.0f}ms"
            )

            axes[idx].set_title(
                title,
                fontsize=8
            )

            axes[idx].axis(
                "off"
            )

        results_summary.append(
            {
                "image": os.path.basename(img_path),
                "source": selected_source,
                "visualization": frame_save_path,
                "n_detections": len(detections),
                "class_counts": class_counts,
                "latency_ms": round(latency_ms, 1),
                "detections": detections
            }
        )

        print(
            f"{os.path.basename(img_path)[:28]:<28} "
            f"Objects:{len(detections):<3} "
            f"{latency_ms:.0f}ms"
        )

    # --------------------------------------------------------
    # Hide unused subplots
    # --------------------------------------------------------
    for i in range(
        len(results_summary),
        len(axes)
    ):

        axes[i].axis(
            "off"
        )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "day3_first_detection.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\n✓ Detection grid saved: {output_path}"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)

    print(
        f"{'Image':<28} "
        f"{'Objects':<10} "
        f"{'Latency'}"
    )

    print("-" * 60)

    total_objects = 0

    for r in results_summary:

        objs = r["n_detections"]
        total_objects += objs

        classes_str = ", ".join(
            [
                f"{k}:{v}"
                for k, v in r["class_counts"].items()
            ]
        )

        print(
            f"{r['image'][:27]:<28} "
            f"{objs:<10} "
            f"{r['latency_ms']:.0f}ms"
        )

        if classes_str:

            print(
                f" └─ {classes_str}"
            )

    print("-" * 60)

    if len(latencies) > 0:

        avg_latency = float(
            np.mean(latencies)
        )

        p50 = float(
            np.percentile(latencies, 50)
        )

        p90 = float(
            np.percentile(latencies, 90)
        )

        p99 = float(
            np.percentile(latencies, 99)
        )

        fps = float(
            1000.0 / avg_latency
        )

    else:

        avg_latency = 0.0
        p50 = 0.0
        p90 = 0.0
        p99 = 0.0
        fps = 0.0

    print(
        f"Total objects detected: {total_objects}"
    )

    print("\nLatency stats:")
    print(
        f" Average: {avg_latency:.1f}ms "
        f"({fps:.2f} FPS)"
    )

    print(
        f" P50: {p50:.1f}ms"
    )

    print(
        f" P90: {p90:.1f}ms"
    )

    print(
        f" P99: {p99:.1f}ms"
    )

    if p99 < BUDGET_MS:

        print(
            f"\n✓ P99 {p99:.0f}ms < "
            f"{BUDGET_MS}ms budget — ON TARGET"
        )

    else:

        print(
            f"\n⚠ P99 {p99:.0f}ms > "
            f"{BUDGET_MS}ms budget — needs optimization"
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = os.path.join(
        OUTPUT_DIR,
        "day3_detection_output.json"
    )

    report = {
        "phase": "Day 3 - Basic Detection Demo",
        "model": MODEL_WEIGHTS,
        "source_dataset": selected_source,
        "total_images": len(results_summary),
        "total_detections": total_objects,
        "avg_latency_ms": round(avg_latency, 1),
        "p50_latency_ms": round(p50, 1),
        "p90_latency_ms": round(p90, 1),
        "p99_latency_ms": round(p99, 1),
        "fps": round(fps, 2),
        "budget_ms": BUDGET_MS,
        "budget_status": (
            "ON_TARGET"
            if p99 < BUDGET_MS
            else "NEEDS_OPTIMIZATION"
        ),
        "results": results_summary
    }

    with open(
        json_path,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print(
        f"\n✓ JSON saved: {json_path}"
    )

    print("\n✅ Day 3 Step 1 Complete!")
    print("Next: Run dataset analysis / adaptive pipeline demo")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
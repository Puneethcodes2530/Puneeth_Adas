"""
Phase 2 Final Demo

NeuroSentinel v3 — Adaptive Detection Pipeline Demo

Shows:
✓ Scene-aware adaptive detection
✓ CLIP-based scene/weather classification
✓ Dynamic YOLO confidence thresholding
✓ VRU highlighting
✓ Geometry-based distance estimation
✓ BDD100K + IDD sample visualization
✓ Summary JSON report

Outputs:
- outputs/phase2_adaptive_detection.png
- outputs/phase2_summary.json
- outputs/phase2_frames/*.png
"""

import os
import sys
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter


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
# IMPORTS
# ============================================================

from perception.detector import AdaptiveDetector


# ============================================================
# PATHS
# ============================================================

PATHS = {
    "BDD100K": r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test",

    "IDD": r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\frontFar",
}


# ============================================================
# CONFIG
# ============================================================

N_IMAGES_PER_DATASET = 8
N_GRID_IMAGES = 6

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs"
)

FRAME_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "phase2_frames"
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
# IMAGE SEARCH
# ============================================================

def find_images(
    folder: str,
    limit: int = 8
):
    """
    Finds images from a folder recursively.

    Supports:
    - jpg
    - jpeg
    - png
    """

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
# SAFE JSON CONVERSION
# ============================================================

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NeuroSentinel v3 — Phase 2 Adaptive Detection Demo")
    print("=" * 70)

    # --------------------------------------------------------
    # Load detector
    # --------------------------------------------------------
    detector = AdaptiveDetector(
        model_weights="yolov8s.pt",
        budget_ms=40.0,
        scene_update_every=15
    )

    # --------------------------------------------------------
    # Collect outputs
    # --------------------------------------------------------
    all_outputs = []

    for dataset_name, folder in PATHS.items():

        images = find_images(
            folder,
            limit=N_IMAGES_PER_DATASET
        )

        if len(images) == 0:

            print(
                f"[WARNING] No images found for {dataset_name}: {folder}"
            )

            continue

        print(
            f"\nProcessing {dataset_name} "
            f"({len(images)} images)..."
        )

        for img_idx, img_path in enumerate(images):

            frame = cv2.imread(
                img_path
            )

            if frame is None:

                print(
                    f"[WARNING] Could not read image: {img_path}"
                )

                continue

            output = detector.process(
                frame
            )

            vis = detector.draw(
                frame,
                output
            )

            # Save individual visualization
            save_name = (
                f"{dataset_name}_{img_idx + 1}_"
                f"{os.path.basename(img_path)}"
            )

            save_path = os.path.join(
                FRAME_OUTPUT_DIR,
                save_name
            )

            cv2.imwrite(
                save_path,
                vis
            )

            all_outputs.append(
                {
                    "dataset": dataset_name,
                    "img_path": img_path,
                    "save_path": save_path,
                    "vis": vis,
                    "output": output
                }
            )

            print(
                f" {os.path.basename(img_path)[:28]:<28} "
                f"Scene:{output.scene.condition:<7} "
                f"Conf:{output.scene.confidence:.2f} "
                f"Objects:{output.n_objects:<3} "
                f"{output.processing_ms:.0f}ms "
                f"{'OK' if output.within_budget else 'SLOW'}"
            )

    # --------------------------------------------------------
    # No output guard
    # --------------------------------------------------------
    if len(all_outputs) == 0:

        print("\nNo outputs generated. Check dataset paths.")

        return

    # ========================================================
    # VISUALIZATION GRID
    # ========================================================

    n = min(
        len(all_outputs),
        N_GRID_IMAGES
    )

    rows = 2
    cols = 3

    fig = plt.figure(
        figsize=(20, 14)
    )

    fig.patch.set_facecolor(
        "#0a0a1a"
    )

    fig.suptitle(
        "NeuroSentinel v3 — Phase 2: Adaptive Detection Pipeline\n"
        "Scene-Aware Dynamic Confidence Thresholding",
        fontsize=15,
        fontweight="bold",
        color="white"
    )

    gs = gridspec.GridSpec(
        rows,
        cols,
        figure=fig,
        hspace=0.30,
        wspace=0.05
    )

    for idx in range(rows * cols):

        row, col = divmod(
            idx,
            cols
        )

        ax = fig.add_subplot(
            gs[row, col]
        )

        if idx >= n:

            ax.axis("off")
            continue

        item = all_outputs[idx]

        vis_rgb = cv2.cvtColor(
            item["vis"],
            cv2.COLOR_BGR2RGB
        )

        ax.imshow(
            vis_rgb
        )

        out = item["output"]

        vru_count = sum(
            1
            for det in out.detections
            if det.is_vru
        )

        title = (
            f"{item['dataset']} | "
            f"{out.scene.condition} "
            f"(conf:{out.scene.confidence:.2f}, "
            f"sev:{out.scene.severity:.2f})\n"
            f"thr:{out.scene.conf_threshold:.2f} | "
            f"pen:{out.scene.confidence_penalty:.2f} | "
            f"{out.n_objects} objects | "
            f"{vru_count} VRU | "
            f"{out.processing_ms:.0f}ms"
        )

        ax.set_title(
            title,
            fontsize=8,
            color="white",
            pad=4
        )

        ax.axis("off")

    plt.tight_layout()

    grid_save_path = os.path.join(
        OUTPUT_DIR,
        "phase2_adaptive_detection.png"
    )

    plt.savefig(
        grid_save_path,
        dpi=150,
        bbox_inches="tight",
        facecolor="#0a0a1a"
    )

    plt.show()

    print(
        f"\n✓ Saved grid: {grid_save_path}"
    )

    # ========================================================
    # STATS SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)

    conditions = [
        item["output"].scene.condition
        for item in all_outputs
    ]

    latencies = [
        item["output"].processing_ms
        for item in all_outputs
    ]

    obj_counts = [
        item["output"].n_objects
        for item in all_outputs
    ]

    within = sum(
        1
        for item in all_outputs
        if item["output"].within_budget
    )

    dataset_counts = Counter(
        item["dataset"]
        for item in all_outputs
    )

    condition_counts = Counter(
        conditions
    )

    avg_latency = safe_float(
        np.mean(latencies)
    )

    p50_latency = safe_float(
        np.percentile(latencies, 50)
    )

    p90_latency = safe_float(
        np.percentile(latencies, 90)
    )

    p99_latency = safe_float(
        np.percentile(latencies, 99)
    )

    avg_objects = safe_float(
        np.mean(obj_counts)
    )

    print(
        f"\nDatasets processed: {dict(dataset_counts)}"
    )

    print(
        f"Conditions detected: {dict(condition_counts)}"
    )

    print(
        f"Avg latency: {avg_latency:.1f}ms"
    )

    print(
        f"P50 latency: {p50_latency:.1f}ms"
    )

    print(
        f"P90 latency: {p90_latency:.1f}ms"
    )

    print(
        f"P99 latency: {p99_latency:.1f}ms"
    )

    print(
        f"Avg objects/frame: {avg_objects:.1f}"
    )

    print(
        f"Within budget: {within}/{len(all_outputs)} frames"
    )

    # ========================================================
    # SAVE JSON SUMMARY
    # ========================================================

    summary = {
        "phase": "Phase 2 - Adaptive Detection Complete",
        "model": "YOLOv8s",
        "scene_module": "CLIPSceneDetector",
        "novel_module": "Scene-aware adaptive confidence thresholding",
        "datasets_processed": dict(dataset_counts),
        "conditions_found": dict(condition_counts),
        "avg_latency_ms": round(avg_latency, 1),
        "p50_latency_ms": round(p50_latency, 1),
        "p90_latency_ms": round(p90_latency, 1),
        "p99_latency_ms": round(p99_latency, 1),
        "avg_objects": round(avg_objects, 1),
        "budget_compliance": f"{within}/{len(all_outputs)}",
        "outputs": [
            {
                "dataset": item["dataset"],
                "image": os.path.basename(
                    item["img_path"]
                ),
                "visualization": item["save_path"],
                "scene": item["output"].scene.condition,
                "scene_confidence": round(
                    item["output"].scene.confidence,
                    3
                ),
                "scene_severity": round(
                    item["output"].scene.severity,
                    3
                ),
                "n_objects": item["output"].n_objects,
                "processing_ms": round(
                    item["output"].processing_ms,
                    1
                ),
                "within_budget": item["output"].within_budget,
                "detections": [
                    {
                        "class": det.class_name,
                        "raw_confidence": det.raw_confidence,
                        "adjusted_confidence": det.adj_confidence,
                        "bbox": det.bbox,
                        "distance_m": det.est_distance_m,
                        "is_vru": det.is_vru
                    }
                    for det in item["output"].detections
                ]
            }
            for item in all_outputs
        ]
    }

    json_save_path = os.path.join(
        OUTPUT_DIR,
        "phase2_summary.json"
    )

    with open(
        json_save_path,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        f"\n✓ Saved JSON: {json_save_path}"
    )

    print("\n🎯 PHASE 2 COMPLETE")
    print("✓ Scene-aware adaptive detection running")
    print("✓ Visual outputs saved")
    print("✓ JSON summary saved")
    print("✓ Ready for Phase 3 / Phase 4 integration")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
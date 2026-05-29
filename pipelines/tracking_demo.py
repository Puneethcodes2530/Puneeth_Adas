"""
NeuroSentinel v3 — Tracking + TTC Demo

Purpose:
- Process an ordered image sequence from one folder
- Run final ADAS perception pipeline
- Show detection + distance + TTC + risk
- Save sample frames
- Save grid visualization
- Optionally save output video

Important:
TTC is meaningful only when images are sequential frames
from the same drive/video. Random images will usually show TTC:N/A.
"""

import os
import sys
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER INPUT — CHANGE THIS PATH ONLY
# ============================================================

IMAGE_FOLDER = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test\testA"

# If IDD has sequential frames, use this instead:
# IMAGE_FOLDER = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\frontFar"

MAX_FRAMES = 60
SAVE_EVERY_N_FRAMES = 10

MODEL_WEIGHTS = "yolov8s.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.35

SAVE_VIDEO = True
SHOW_GRID = True


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

from perception.tracker import Phase4TrackerPipeline
from perception.depth_estimator import DepthEstimatorDA


# ============================================================
# OUTPUT DIRS
# ============================================================

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs",
    "phase4_tracking"
)

FRAME_DIR = os.path.join(
    OUTPUT_DIR,
    "frames"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    FRAME_DIR,
    exist_ok=True
)


# ============================================================
# DEPTH ENGINE
# ============================================================

print("[INFO] Loading depth engine...")

depth_engine = DepthEstimatorDA()


def depth_fn(frame):
    depth_output = depth_engine.estimate(frame)
    return depth_output.depth_map


# ============================================================
# IMAGE LOADER
# ============================================================

def find_images(folder, limit=60):

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Image folder not found: {folder}"
        )

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
# DEPTH VISUALIZATION
# ============================================================

def make_depth_vis(depth_map, target_shape):

    h, w = target_shape[:2]

    if depth_map is None:

        blank = np.zeros(
            (h, w, 3),
            dtype=np.uint8
        )

        cv2.putText(
            blank,
            "No depth map",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        return blank

    d = depth_map.copy()

    d = (
        (d - d.min()) /
        (d.max() - d.min() + 1e-6)
    )

    d = (
        d * 255
    ).astype(np.uint8)

    d_col = cv2.applyColorMap(
        d,
        cv2.COLORMAP_MAGMA
    )

    if d_col.shape[:2] != (h, w):

        d_col = cv2.resize(
            d_col,
            (w, h)
        )

    return d_col


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NeuroSentinel v3 — Tracking + TTC Demo")
    print("=" * 70)

    print("Image folder:", IMAGE_FOLDER)

    images = find_images(
        IMAGE_FOLDER,
        limit=MAX_FRAMES
    )

    if len(images) == 0:
        print("[ERROR] No images found.")
        return

    print(f"Processing {len(images)} frames")

    # --------------------------------------------------------
    # Load pipeline
    # --------------------------------------------------------

    pipe = Phase4TrackerPipeline(
        model_weights=MODEL_WEIGHTS,
        depth_fn=depth_fn,
        fps=30,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        use_tracker=True
    )

    sample_frames = []
    latencies = []
    object_counts = []

    video_writer = None

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    for idx, img_path in enumerate(images):

        frame = cv2.imread(
            img_path
        )

        if frame is None:
            print("[WARNING] Could not read:", img_path)
            continue

        objects, depth_map, latency = pipe.process(
            frame
        )

        vis = pipe.draw(
            frame,
            objects,
            latency
        )

        latencies.append(
            latency
        )

        object_counts.append(
            len(objects)
        )

        # ----------------------------------------------------
        # Save video
        # ----------------------------------------------------

        if SAVE_VIDEO:

            if video_writer is None:

                h, w = vis.shape[:2]

                video_path = os.path.join(
                    OUTPUT_DIR,
                    "phase4_tracking_demo.mp4"
                )

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                video_writer = cv2.VideoWriter(
                    video_path,
                    fourcc,
                    10,
                    (w, h)
                )

            video_writer.write(
                vis
            )

        # ----------------------------------------------------
        # Save sample frames
        # ----------------------------------------------------

        if idx % SAVE_EVERY_N_FRAMES == 0:

            frame_save_path = os.path.join(
                FRAME_DIR,
                f"frame_{idx:04d}.png"
            )

            cv2.imwrite(
                frame_save_path,
                vis
            )

            depth_vis = make_depth_vis(
                depth_map,
                frame.shape
            )

            combined = np.hstack(
                [
                    vis,
                    depth_vis
                ]
            )

            combined_save_path = os.path.join(
                FRAME_DIR,
                f"frame_{idx:04d}_combined.png"
            )

            cv2.imwrite(
                combined_save_path,
                combined
            )

            sample_frames.append(
                (
                    idx,
                    vis,
                    len(objects),
                    latency
                )
            )

        # ----------------------------------------------------
        # Console print
        # ----------------------------------------------------

        print(
            f"[{idx + 1:03d}/{len(images)}] "
            f"{os.path.basename(img_path)[:28]:<28} "
            f"objects:{len(objects):<3} "
            f"latency:{latency:.0f}ms"
        )

        for obj in objects[:5]:

            print(
                f"   ID:{obj.track_id:<5} "
                f"{obj.class_name:<12} "
                f"dist:{obj.distance_m:<6.1f}m "
                f"TTC:{obj.ttc.value if obj.ttc.value is not None else 'N/A':<6} "
                f"risk:{obj.risk:<9} "
                f"score:{obj.risk_score:.2f}"
            )

    if video_writer is not None:
        video_writer.release()
        print("\n✓ Video saved:", video_path)

    # ========================================================
    # GRID VISUALIZATION
    # ========================================================

    if SHOW_GRID and len(sample_frames) > 0:

        n = min(
            len(sample_frames),
            6
        )

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(20, 12)
        )

        fig.patch.set_facecolor(
            "#0a0a1a"
        )

        fig.suptitle(
            "NeuroSentinel v3 — Phase 4 Tracking + TTC + Risk Demo",
            fontsize=15,
            fontweight="bold",
            color="white"
        )

        axes = axes.flatten()

        for i in range(6):

            ax = axes[i]

            if i >= n:

                ax.axis("off")
                continue

            frame_idx, vis, n_obj, latency = sample_frames[i]

            ax.imshow(
                cv2.cvtColor(
                    vis,
                    cv2.COLOR_BGR2RGB
                )
            )

            ax.set_title(
                f"Frame {frame_idx} | Objects:{n_obj} | {latency:.0f}ms",
                color="white",
                fontsize=9
            )

            ax.axis("off")

        plt.tight_layout()

        grid_path = os.path.join(
            OUTPUT_DIR,
            "phase4_tracking_grid.png"
        )

        plt.savefig(
            grid_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="#0a0a1a"
        )

        plt.show()

        print("✓ Grid saved:", grid_path)

    # ========================================================
    # SUMMARY
    # ========================================================

    if len(latencies) > 0:

        print("\n" + "=" * 70)
        print("PHASE 4 TRACKING SUMMARY")
        print("=" * 70)

        print(
            f"Frames processed: {len(latencies)}"
        )

        print(
            f"Avg latency: {np.mean(latencies):.1f}ms"
        )

        print(
            f"P50 latency: {np.percentile(latencies, 50):.1f}ms"
        )

        print(
            f"P90 latency: {np.percentile(latencies, 90):.1f}ms"
        )

        print(
            f"P99 latency: {np.percentile(latencies, 99):.1f}ms"
        )

        print(
            f"FPS: {1000 / np.mean(latencies):.2f}"
        )

        print(
            f"Avg objects/frame: {np.mean(object_counts):.1f}"
        )

    print("\n🎯 Phase 4 Tracking Demo Complete")
    print("Outputs saved in:", OUTPUT_DIR)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()

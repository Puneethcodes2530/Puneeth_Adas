"""
NeuroSentinel v3 — Single Image Demo

Purpose:
- Give one image path inside this file
- Run final ADAS perception pipeline
- Show detection + distance + TTC + risk
- Show depth map
- Save final demo output

How to run:
python pipelines/single_image_demo.py
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER INPUT — CHANGE IMAGE PATH HERE ONLY
# ============================================================

IMAGE_PATH = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI\data_object_image_2\training\image_2\007444.png"

# Stable demo settings
MODEL_WEIGHTS = "yolov8s.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.35
SHOW_OUTPUT = True


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
# OUTPUT PATHS
# ============================================================

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs",
    "demo"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEPTH ENGINE
# ============================================================

print("[INFO] Loading depth engine...")

depth_engine = DepthEstimatorDA()


def depth_fn(frame):
    """
    Function passed into tracker pipeline.
    Returns only depth_map.
    """

    depth_output = depth_engine.estimate(
        frame
    )

    return depth_output.depth_map


# ============================================================
# DEPTH VISUALIZATION
# ============================================================

def make_depth_vis(
    depth_map,
    target_shape=None
):
    """
    Converts depth map to colored visualization.
    """

    if depth_map is None:

        if target_shape is None:
            return None

        h, w = target_shape[:2]

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

    if target_shape is not None:

        h, w = target_shape[:2]

        if d_col.shape[:2] != (h, w):

            d_col = cv2.resize(
                d_col,
                (w, h)
            )

    return d_col


# ============================================================
# PIPELINE LOADER
# ============================================================

def load_pipeline():
    """
    Loads final Phase 4 ADAS pipeline.
    """

    print("[INFO] Loading final ADAS pipeline...")

    pipe = Phase4TrackerPipeline(
        model_weights=MODEL_WEIGHTS,
        depth_fn=depth_fn,
        fps=30,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        use_tracker=True
    )

    print("[INFO] Pipeline loaded.")

    return pipe


# ============================================================
# MAIN SINGLE IMAGE DEMO
# ============================================================

def main():

    image_path = IMAGE_PATH

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        raise ValueError(
            f"Could not read image: {image_path}"
        )

    print("\n" + "=" * 70)
    print("NeuroSentinel v3 — Single Image Demo")
    print("=" * 70)
    print("Image:", image_path)
    print("Model:", MODEL_WEIGHTS)
    print("Image size:", frame.shape)

    pipe = load_pipeline()

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    objects, depth_map, latency = pipe.process(
        frame
    )

    print("\nDetected objects:", len(objects))
    print(f"Latency: {latency:.1f} ms")

    for obj in objects:

        print(
            f"ID:{obj.track_id:<5} "
            f"{obj.class_name:<12} "
            f"conf:{obj.confidence:<5.2f} "
            f"dist:{obj.distance_m:<6.1f}m "
            f"TTC:{obj.ttc.value} "
            f"risk:{obj.risk:<9} "
            f"score:{obj.risk_score:.2f}"
        )

    # --------------------------------------------------------
    # Draw annotated output
    # --------------------------------------------------------

    vis = pipe.draw(
        frame,
        objects,
        latency
    )

    depth_vis = make_depth_vis(
        depth_map,
        target_shape=frame.shape
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    annotated_path = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_annotated.png"
    )

    depth_path = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_depth.png"
    )

    combined_path = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_combined.png"
    )

    cv2.imwrite(
        annotated_path,
        vis
    )

    if depth_vis is not None:

        cv2.imwrite(
            depth_path,
            depth_vis
        )

        combined = np.hstack(
            [
                vis,
                depth_vis
            ]
        )

    else:

        combined = vis

    cv2.imwrite(
        combined_path,
        combined
    )

    print("\nSaved:")
    print("Annotated:", annotated_path)
    print("Depth:", depth_path)
    print("Combined:", combined_path)

    # --------------------------------------------------------
    # Show output
    # --------------------------------------------------------

    if SHOW_OUTPUT:

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(22, 7)
        )

        fig.suptitle(
            "NeuroSentinel v3 — Single Image ADAS Demo",
            fontsize=16,
            fontweight="bold"
        )

        axes[0].imshow(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )

        axes[0].set_title(
            "Original Image"
        )

        axes[0].axis(
            "off"
        )

        axes[1].imshow(
            cv2.cvtColor(
                vis,
                cv2.COLOR_BGR2RGB
            )
        )

        axes[1].set_title(
            "Detection + Distance + TTC + Risk"
        )

        axes[1].axis(
            "off"
        )

        axes[2].imshow(
            cv2.cvtColor(
                depth_vis,
                cv2.COLOR_BGR2RGB
            )
        )

        axes[2].set_title(
            "Depth Map"
        )

        axes[2].axis(
            "off"
        )

        plt.tight_layout()
        plt.show()

    print("\n✅ Single image demo complete.")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()

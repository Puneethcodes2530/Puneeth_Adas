# ============================================================
# FILE: day4_depth_test.py
# ============================================================

"""
Phase 3 Step 1 — Depth Estimation Test
Relative depth-based distance estimation
"""

import cv2
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('.'))

from perception.depth_estimator_da import DepthEstimatorDA
from perception.adaptive_detector import AdaptiveDetector
from perception.clip_scene_detector import CLIPSceneDetector


# ── PATHS ──────────────────────────────────────────────────

BDD_PATH = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI\data_object_image_2\testing\image_2\007510.png"

IDD_PATH = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\highquality_16k\HYD-2018-04-26_12-40-01\0006527.jpg"


# ── IMAGE LIST ─────────────────────────────────────────────

images = [
    BDD_PATH,
    IDD_PATH
]

if len(images) == 0:

    print("No valid images found.")
    exit()


# ── LOAD MODELS ────────────────────────────────────────────

print("Loading models...")

depth_est = DepthEstimatorDA()

detector = AdaptiveDetector(
    'yolov8x.pt'
)

clip_detector = CLIPSceneDetector()


# ── OUTPUT DIR ─────────────────────────────────────────────

os.makedirs(
    "outputs",
    exist_ok=True
)


# ── PLOT SETUP ─────────────────────────────────────────────

fig, axes = plt.subplots(
    len(images),
    3,
    squeeze=False,
    figsize=(18, len(images) * 5)
)

fig.patch.set_facecolor('#0a0a1a')

fig.suptitle(
    'NeuroSentinel v3 — Phase 3: Detection + Depth\n'
    'Geometry-guided monocular distance estimation',
    fontsize=14,
    fontweight='bold',
    color='white'
)


# ── PROCESS IMAGES ─────────────────────────────────────────

for idx, img_path in enumerate(images):

    print(f"\nProcessing: {os.path.basename(img_path)}")

    frame = cv2.imread(img_path)

    if frame is None:

        print(f"[ERROR] Failed to load image: {img_path}")
        continue

    # ── CLIP SCENE ANALYSIS ────────────────────────────────

    clip_condition, clip_conf = clip_detector.analyze(frame)

    print(
        f"[CLIP] {clip_condition} "
        f"(conf: {clip_conf:.2f})"
    )

    # ── OBJECT DETECTION ───────────────────────────────────

    det_output = detector.process(frame)

    det_vis = detector.draw(
        frame,
        det_output
    )

    # ── DEPTH ESTIMATION ───────────────────────────────────

    depth_out = depth_est.estimate(frame)

    depth_vis = depth_est.visualize(
        frame,
        depth_out
    )

    # ── DISTANCE VISUALIZATION ─────────────────────────────

    det_with_depth = det_vis.copy()

    detection_summaries = []

    for det in det_output.detections:

        depth_out.current_class = det.class_name

        depth_sample = depth_out.sample_at_bbox(
            det.bbox
        )

        real_dist = depth_sample['distance_m']

        confidence = depth_sample['confidence']

        # ----------------------------------------------------
        # DISTANCE ZONES
        # ----------------------------------------------------

        zone = (
            "NEAR" if real_dist < 10 else
            "MED" if real_dist < 25 else
            "FAR"
        )

        # ----------------------------------------------------
        # SAVE SUMMARY
        # ----------------------------------------------------

        detection_summaries.append(
            f"{det.class_name}:{real_dist:.1f}m"
        )

        # ----------------------------------------------------
        # LABEL DRAWING
        # ----------------------------------------------------

        x1, y1 = det.bbox[0], det.bbox[1]

        label = (
            f"{zone} "
            f"({real_dist:.1f}m)"
        )

        cv2.putText(
            det_with_depth,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # CONSOLE OUTPUT
        # ----------------------------------------------------

        print(
            f"{det.class_name:<15} "
            f"{real_dist:>6.1f}m "
            f"{zone:<4} "
            f"conf:{confidence:.2f}"
        )

    # ── PLOTTING ───────────────────────────────────────────

    row = axes[idx]

    # --------------------------------------------------------
    # DETECTION VIEW
    # --------------------------------------------------------

    row[0].imshow(
        cv2.cvtColor(
            det_vis,
            cv2.COLOR_BGR2RGB
        )
    )

    row[0].set_title(
        f'Detection | {det_output.scene.condition} | '
        f'CLIP: {clip_condition} | '
        f'{det_output.n_objects} objects',
        color='white',
        fontsize=8
    )

    row[0].axis('off')

    # --------------------------------------------------------
    # DEPTH VIEW
    # --------------------------------------------------------

    depth_right = depth_vis[
        :,
        depth_vis.shape[1] // 2:
    ]

    row[1].imshow(
        cv2.cvtColor(
            depth_right,
            cv2.COLOR_BGR2RGB
        )
    )

    row[1].set_title(
        f'Depth Map | {depth_out.model_name} | '
        f'{depth_out.processing_ms:.0f}ms',
        color='white',
        fontsize=8
    )

    row[1].axis('off')

    # --------------------------------------------------------
    # DISTANCE VIEW
    # --------------------------------------------------------

    row[2].imshow(
        cv2.cvtColor(
            det_with_depth,
            cv2.COLOR_BGR2RGB
        )
    )

    row[2].set_title(
        'Distance Estimation\n' +
        ' | '.join(
            detection_summaries[:4]
        ),
        color='cyan',
        fontsize=8
    )

    row[2].axis('off')


# ── SAVE RESULTS ───────────────────────────────────────────

plt.tight_layout()

save_path = (
    "outputs/"
    "phase3_depth_detection.png"
)

plt.savefig(
    save_path,
    dpi=150,
    bbox_inches='tight',
    facecolor='#0a0a1a'
)

plt.show()

print(f"\n✓ Saved: {save_path}")

print("\n🎯 Phase 3 Step 1 complete")

print("Next: TTC Engine + Risk Scoring")
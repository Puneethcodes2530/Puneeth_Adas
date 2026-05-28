import cv2
import os
import glob
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('.'))

from perception.tracker import (
    Phase4TrackerPipeline
)

# ============================================================
# IMAGE PATH
# ============================================================

IMAGE_FOLDER = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test"

# ============================================================
# LOAD PIPELINE
# ============================================================

pipe = Phase4TrackerPipeline()

images = sorted(
    glob.glob(f"{IMAGE_FOLDER}/*.jpg")
)[:60]

if len(images) == 0:

    print("No images found")

    exit()

print(f"Processing {len(images)} frames")

# ============================================================
# OUTPUT
# ============================================================

os.makedirs(
    "outputs/phase4",
    exist_ok=True
)

sample_frames = []

# ============================================================
# LOOP
# ============================================================

for idx, img_path in enumerate(images):

    frame = cv2.imread(img_path)

    if frame is None:
        continue

    objects, depth_map, latency = pipe.process(
        frame
    )

    vis = pipe.draw(

        frame,

        objects,

        latency
    )

    # --------------------------------------------------------
    # SAVE SAMPLE
    # --------------------------------------------------------

    if idx % 10 == 0:

        sample_frames.append(

            (
                idx,

                vis.copy()
            )
        )

    # --------------------------------------------------------
    # CONSOLE
    # --------------------------------------------------------

    print(

        f"[{idx+1}/{len(images)}] "

        f"Objects:{len(objects)} "

        f"Latency:{latency:.0f}ms"
    )

# ============================================================
# GRID
# ============================================================

fig, axes = plt.subplots(

    2,

    3,

    figsize=(18, 10)
)

fig.patch.set_facecolor('#0a0a1a')

fig.suptitle(

    'NeuroSentinel v3 — Phase 4\n'

    'Tracking + TTC + Risk',

    fontsize=14,

    fontweight='bold',

    color='white'
)

axes = axes.flatten()

for i, (frame_idx, vis) in enumerate(
    sample_frames[:6]
):

    axes[i].imshow(
        cv2.cvtColor(
            vis,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[i].set_title(

        f"Frame {frame_idx}",

        color='white'
    )

    axes[i].axis('off')

for i in range(
    len(sample_frames),
    6
):

    axes[i].set_visible(False)

plt.tight_layout()

save_path = (
    "outputs/phase4/"
    "phase4_tracking_grid.png"
)

plt.savefig(

    save_path,

    dpi=150,

    bbox_inches='tight',

    facecolor='#0a0a1a'
)

plt.show()

print(f"\n✓ Saved: {save_path}")

print("\n🎯 Phase 4 COMPLETE")
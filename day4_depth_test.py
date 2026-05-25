"""
Phase 3 Step 1 — Depth Estimation Test
"Relative depth-based distance estimation"

"""
import cv2
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from perception.depth_estimator import DepthEstimator
from perception.adaptive_detector import AdaptiveDetector

# ── PATHS ──────────────────────────────────────────────────
BDD_PATH = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test\e9cdc338-ab4824c7.jpg"
IDD_PATH = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\frontFar\BLR-2018-04-26_13-40-01_frontFar\000882_r.jpg"

# ── Load models ────────────────────────────────────────────
print("Loading models...")
depth_est = DepthEstimator()
detector = AdaptiveDetector('yolov8s.pt')

# ── Get test images ────────────────────────────────────────
images = [BDD_PATH, IDD_PATH]

if not images:
    print("Update paths above")
    exit()

# ── Run detection + depth ──────────────────────────────────
os.makedirs("outputs", exist_ok=True)
fig, axes = plt.subplots(len(images), 3,
                          figsize=(18, len(images)*5))
fig.patch.set_facecolor('#0a0a1a')
fig.suptitle(
    'NeuroSentinel v3 — Phase 3: Detection + Depth\n'
    'Relative depth-based distance estimation',
    fontsize=14, fontweight='bold', color='white'
)

for idx, img_path in enumerate(images):
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    # Detection
    det_output = detector.process(frame)
    det_vis = detector.draw(frame, det_output)

    # Depth
    depth_out = depth_est.estimate(frame)
    depth_vis = depth_est.visualize(frame, depth_out)

    # Draw depth-sampled distances on detection frame
    det_with_depth = det_vis.copy()
    for det in det_output.detections:
        depth_sample = depth_out.sample_at_bbox(det.bbox)
        real_dist = depth_sample['distance_m']
        conf = depth_sample['confidence']

        x1, y1 = det.bbox[0], det.bbox[1]
        dist_label = f"D:{real_dist:.1f}m"
        cv2.putText(det_with_depth, dist_label,
                    (x1, y1-22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)

    # Plot
    row = axes[idx] if len(images) > 1 else axes

    # Col 0: original detection
    row[0].imshow(
        cv2.cvtColor(det_vis, cv2.COLOR_BGR2RGB))
    row[0].set_title(
        f'Detection | {det_output.scene.condition} | '
        f'{det_output.n_objects} objects',
        color='white', fontsize=8)
    row[0].axis('off')

    # Col 1: depth map
    depth_right = depth_vis[:, depth_vis.shape[1]//2:]
    row[1].imshow(
        cv2.cvtColor(depth_right, cv2.COLOR_BGR2RGB))
    row[1].set_title(
        f'Depth Map | {depth_out.model_name} | '
        f'{depth_out.processing_ms:.0f}ms',
        color='white', fontsize=8)
    row[1].axis('off')

    # Col 2: detection + real distances
    row[2].imshow(
        cv2.cvtColor(det_with_depth, cv2.COLOR_BGR2RGB))

    # Print distance summary
    dist_summary = []
    for det in det_output.detections[:3]:
        d = depth_out.sample_at_bbox(det.bbox)
        dist_summary.append(
            f"{det.class_name}:{d['distance_m']:.1f}m")
    row[2].set_title(
        'With Real Distances\n' +
        ' '.join(dist_summary),
        color='cyan', fontsize=8)
    row[2].axis('off')

    print(f"\n{os.path.basename(img_path)}:")
    for det in det_output.detections:
        d = depth_out.sample_at_bbox(det.bbox)
        zone = ('NEAR' if d['distance_m'] < 10
                else 'MED' if d['distance_m'] < 30
                else 'FAR')
        print(f" {det.class_name:<15} "
              f"{d['distance_m']:>6.1f}m "
              f"{zone} "
              f"conf:{d['confidence']:.2f}")

plt.tight_layout()
plt.savefig("outputs/phase3_depth_detection.png",
            dpi=150, bbox_inches='tight',
            facecolor='#0a0a1a')
plt.show()
print("\n✓ Saved: outputs/phase3_depth_detection.png")
print("\n🎯 Phase 3 Step 1 complete")
print("Next: TTC Engine + Risk Scoring")
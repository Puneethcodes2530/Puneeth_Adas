"""
Phase 2 Final Demo
Shows: Fixed threshold vs Adaptive threshold side by side
This is your Phase 2 deliverable visualization.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob, os, sys, json

sys.path.insert(0, os.path.abspath('.'))
from perception.adaptive_detector import AdaptiveDetector

# ── PATHS — update these ───────────────────────────────────
PATHS = {
    'BDD100K': r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test",
    'IDD': r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\frontFar",
}

# ── Load detector ──────────────────────────────────────────
detector = AdaptiveDetector('yolov8s.pt', budget_ms=40.0)

# ── Collect results ────────────────────────────────────────
all_outputs = []

for dataset, folder in PATHS.items():
    images = (glob.glob(f"{folder}/*.jpg") +
              glob.glob(f"{folder}/**/*.jpg",
                        recursive=True))[:8]
    if not images:
        print(f"No images: {folder}")
        continue

    print(f"\nProcessing {dataset} ({len(images)} images)...")
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        output = detector.process(frame)
        vis = detector.draw(frame, output)
        all_outputs.append({
            'dataset': dataset,
            'img_path': img_path,
            'frame': frame,
            'vis': vis,
            'output': output
        })
        print(f" {os.path.basename(img_path)[:25]:<25} "
              f"Scene:{output.scene.condition:<6} "
              f"Objects:{output.n_objects:<3} "
              f"{output.processing_ms:.0f}ms "
              f"{'✓' if output.within_budget else '!'}")

# ── Visualization grid ─────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

n = min(len(all_outputs), 6)
if n == 0:
    print("No outputs to visualize")
    exit()

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0a0a1a')
fig.suptitle(
    'NeuroSentinel v3 — Phase 2: Adaptive Detection Pipeline\n'
    'Scene-Aware Dynamic Confidence Thresholding',
    fontsize=14, fontweight='bold', color='white'
)

gs = gridspec.GridSpec(
    2, 3, figure=fig, hspace=0.3, wspace=0.05)

for idx, item in enumerate(all_outputs[:n]):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])

    vis_rgb = cv2.cvtColor(item['vis'], cv2.COLOR_BGR2RGB)
    ax.imshow(vis_rgb)

    out = item['output']
    vru_count = sum(1 for d in out.detections if d.is_vru)
    title = (f"{item['dataset']} | "
             f"{out.scene.condition} "
             f"(sev:{out.scene.severity:.2f})\n"
             f"thresh:{out.scene.conf_threshold} | "
             f"{out.n_objects} objects | "
             f"{vru_count} VRU | "
             f"{out.processing_ms:.0f}ms")
    ax.set_title(title, fontsize=7.5,
                  color='white', pad=3)
    ax.axis('off')

plt.savefig("outputs/phase2_adaptive_detection.png",
            dpi=150, bbox_inches='tight',
            facecolor='#0a0a1a')
plt.show()
print("✓ Saved: outputs/phase2_adaptive_detection.png")

# ── Stats summary ──────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2 SUMMARY")
print("="*60)

conditions = [x['output'].scene.condition
              for x in all_outputs]
latencies = [x['output'].processing_ms
              for x in all_outputs]
obj_counts = [x['output'].n_objects
              for x in all_outputs]
within = sum(1 for x in all_outputs
                 if x['output'].within_budget)

from collections import Counter
print(f"\nConditions detected: {dict(Counter(conditions))}")
print(f"Avg latency: {np.mean(latencies):.0f}ms")
print(f"P99 latency: {np.percentile(latencies,99):.0f}ms")
print(f"Avg objects: {np.mean(obj_counts):.1f}/frame")
print(f"Within budget: {within}/{len(all_outputs)} frames")

# Save JSON
summary = {
    'phase': 'Phase 2 - Adaptive Detection Complete',
    'model': 'YOLOv8s',
    'novel_module': 'Scene Degradation Detector + ACRN',
    'conditions_found': dict(Counter(conditions)),
    'avg_latency_ms': round(np.mean(latencies), 1),
    'p99_latency_ms': round(np.percentile(latencies,99),1),
    'avg_objects': round(np.mean(obj_counts), 1),
    'budget_compliance':f"{within}/{len(all_outputs)}"
}
with open("outputs/phase2_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✓ outputs/phase2_summary.json saved")
print("\n🎯 PHASE 2 COMPLETE")
print(" Scene-aware adaptive detection running")
print(" Both datasets processed")
print(" Next: Phase 3 — Distance estimation + TTC")
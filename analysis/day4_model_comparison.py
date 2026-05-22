
"""
Benchmark YOLOv8 model sizes for ADAS speed-accuracy tradeoff
"""
import torch
from ultralytics import YOLO
import cv2, glob, time, json
import numpy as np
import matplotlib.pyplot as plt
import os

# Your BDD path
BDD_PATH = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test"

images = glob.glob(f"{BDD_PATH}/**/*.jpg", recursive=True)[:50]
if not images:
    print("Update BDD_PATH")
    exit()

models_to_test = {
    'YOLOv8n': 'yolov8n.pt', # nano - fastest
    'YOLOv8s': 'yolov8s.pt', # small
    'YOLOv8m': 'yolov8m.pt', # medium
    'YOLOv8x': 'yolov8x.pt', # extra large - slowest
}

results = {}

for model_name, weights in models_to_test.items():
    print(f"\nTesting {model_name}...")
    model = YOLO(weights)
    model.to('cuda')
    model.predict(np.zeros((640,640,3), dtype=np.uint8), device=0)

    latencies = []
    obj_counts = []

    for img_path in images:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        res = model(img_path, conf=0.35, device=0, verbose=False)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        latencies.append(ms)
        obj_counts.append(len(res[0].boxes))

    results[model_name] = {
        'p50_ms': round(np.percentile(latencies, 50), 1),
        'p99_ms': round(np.percentile(latencies, 99), 1),
        'fps': round(1000 / np.mean(latencies), 1),
        'avg_objects': round(np.mean(obj_counts), 1),
    }

    p99 = results[model_name]['p99_ms']
    fps = results[model_name]['fps']
    obj = results[model_name]['avg_objects']
    budget_ok = "✓" if p99 <= 40 else "✗"
    print(f" P99: {p99}ms FPS: {fps} "
          f"Avg objects: {obj} Budget: {budget_ok}")

# ── Plot ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('YOLOv8 Model Size vs Performance\n'
             'For ADAS 40ms Budget Selection',
             fontsize=13, fontweight='bold')

names = list(results.keys())
p99s = [results[m]['p99_ms'] for m in names]
fpss = [results[m]['fps'] for m in names]
objs = [results[m]['avg_objects'] for m in names]

colors = ['#2ecc71' if p <= 40 else '#e74c3c' for p in p99s]

# P99 Latency
axes[0].bar(names, p99s, color=colors)
axes[0].axhline(y=40, color='orange', linestyle='--',
                label='40ms budget')
axes[0].set_title('P99 Latency (lower = better)')
axes[0].set_ylabel('milliseconds')
axes[0].legend()
for i, v in enumerate(p99s):
    axes[0].text(i, v+1, f'{v}ms', ha='center',
                 fontweight='bold')

# FPS
axes[1].bar(names, fpss, color=colors)
axes[1].axhline(y=25, color='orange', linestyle='--',
                label='25 FPS minimum')
axes[1].set_title('Frames Per Second (higher = better)')
axes[1].set_ylabel('FPS')
axes[1].legend()
for i, v in enumerate(fpss):
    axes[1].text(i, v+0.3, f'{v}', ha='center',
                 fontweight='bold')

# Objects detected
axes[2].bar(names, objs, color='#3498db')
axes[2].set_title('Avg Objects Detected\n(proxy for accuracy)')
axes[2].set_ylabel('objects per frame')
for i, v in enumerate(objs):
    axes[2].text(i, v+0.1, f'{v}', ha='center',
                 fontweight='bold')

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/day3_model_comparison.png",
            dpi=150, bbox_inches='tight')
plt.show()

# ── Recommendation ─────────────────────────────────────────
print("\n" + "="*55)
print("RECOMMENDATION FOR NEUROSENTINEL v3")
print("="*55)

within_budget = [m for m in names if results[m]['p99_ms'] <= 40]
over_budget = [m for m in names if results[m]['p99_ms'] > 40]

print(f"\nWithin 40ms budget: {within_budget}")
print(f"Over budget: {over_budget}")

if within_budget:
    best = max(within_budget,
               key=lambda m: results[m]['avg_objects'])
    print(f"\nRecommended model: {best}")
    print(f" P99: {results[best]['p99_ms']}ms")
    print(f" FPS: {results[best]['fps']}")
    print(f" This becomes your baseline for all phases")

with open("outputs/day3_model_comparison.json", 'w') as f:
    json.dump(results, f, indent=2)
print("\n✓ Saved: outputs/day3_model_comparison.json")
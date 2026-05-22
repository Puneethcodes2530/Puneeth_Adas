
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import glob
import os
import time
import yaml


# ── Load config ────────────────────────────────────────────
with open("configs/paths.yaml") as f:
    config = yaml.safe_load(f)


# ── Load YOLOv8 (downloads 22MB automatically) ────────────
print("Loading YOLOv8x pretrained...")
model = YOLO('yolov8x.pt')
print(f"✓ Model loaded")
print(f" Classes: {list(model.names.values())[:10]}...")


# ── Find available images ──────────────────────────────────
def find_images(folder, limit=5):
    images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        images.extend(glob.glob(f"{folder}/{ext}"))
        images.extend(glob.glob(f"{folder}/**/{ext}", recursive=True))
    return images[:limit]


# Try datasets in priority order
image_sources = [
    ("KITTI", config['datasets']['kitti']['training_images']),
    ("BDD100K", config['datasets']['bdd100k']['images_testA']),
    ("UA-DETRAC", config['datasets']['ua_detrac']['train_images']),
    ("IDD", config['datasets']['idd']['images']),
]

selected_images = []
selected_source = None

for source_name, source_path in image_sources:
    imgs = find_images(source_path, limit=6)
    if imgs:
        selected_images = imgs
        selected_source = source_name
        print(f"\n✓ Using {source_name}: found {len(imgs)} images")
        break

if not selected_images:
    print("No images found — check your paths in configs/paths.yaml")
    exit()


# ── Run detection on multiple images ──────────────────────
print(f"\nRunning detection on {len(selected_images)} images...")

ADAS_CLASSES = {
    'person': (255, 50, 50), # Red
    'bicycle': (255, 165, 0), # Orange
    'motorcycle': (255, 140, 0), # Orange
    'car': (50, 205, 50), # Green
    'truck': (0, 128, 255), # Blue
    'bus': (128, 0, 255), # Purple
    'traffic light': (255, 255, 0), # Yellow
    'stop sign': (255, 0, 128), # Pink
}

results_summary = []
latencies = []

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/frames", exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'NeuroSentinel v3 — First Detection\nDataset: {selected_source}',
             fontsize=14, fontweight='bold')
axes = axes.flatten()

for idx, img_path in enumerate(selected_images[:6]):

    # Time the inference
    t_start = time.perf_counter()
    results = model(img_path, conf=0.35, verbose=False)
    latency_ms = (time.perf_counter() - t_start) * 1000
    latencies.append(latency_ms)

    result = results[0]

    # Load and draw
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Count by class
    class_counts = {}
    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_name = model.names[int(box.cls[0])]

        # Only draw ADAS-relevant classes
        if cls_name not in ADAS_CLASSES:
            continue

        color = ADAS_CLASSES[cls_name]
        cv2.rectangle(img_rgb, (x1,y1), (x2,y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        label_size = cv2.getTextSize(label,
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.45, 1)[0]
        cv2.rectangle(img_rgb,
                      (x1, y1 - label_size[1] - 6),
                      (x1 + label_size[0], y1),
                      color, -1)
        cv2.putText(img_rgb, label, (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0,0,0), 1)

        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        detections.append({
            'class': cls_name,
            'confidence': round(conf, 3),
            'bbox': [x1, y1, x2, y2],
            'bbox_width_px': x2-x1,
            'bbox_height_px': y2-y1
        })

    # Add to subplot
    if idx < len(axes):
        axes[idx].imshow(img_rgb)
        title_parts = [f"{cls}: {cnt}"
                       for cls, cnt in class_counts.items()]
        axes[idx].set_title(
            f"{os.path.basename(img_path)[:20]}\n"
            f"{', '.join(title_parts) if title_parts else 'No ADAS objects'}\n"
            f"Latency: {latency_ms:.0f}ms",
            fontsize=8
        )
        axes[idx].axis('off')

    results_summary.append({
        'image': os.path.basename(img_path),
        'source': selected_source,
        'n_detections': len(detections),
        'class_counts': class_counts,
        'latency_ms': round(latency_ms, 1),
        'detections': detections
    })

# Hide unused subplots
for i in range(len(selected_images), 6):
    axes[i].axis('off')

plt.tight_layout()
output_path = "outputs/day3_first_detection.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✓ Detection grid saved: {output_path}")


# ── Print summary table ────────────────────────────────────
print("\n" + "="*60)
print("DETECTION SUMMARY")
print("="*60)
print(f"{'Image':<25} {'Objects':<10} {'Latency'}")
print("-"*50)

total_objects = 0
for r in results_summary:
    objs = r['n_detections']
    total_objects += objs
    classes_str = ", ".join([f"{k}:{v}"
                              for k,v in r['class_counts'].items()])
    print(f"{r['image'][:24]:<25} {objs:<10} {r['latency_ms']:.0f}ms")
    if classes_str:
        print(f" └─ {classes_str}")

print("-"*50)
print(f"Total objects detected: {total_objects}")
print(f"\nLatency stats:")
print(f" Average: {np.mean(latencies):.0f}ms "
      f"({1000/np.mean(latencies):.1f} FPS)")
print(f" P50: {np.percentile(latencies, 50):.0f}ms")
print(f" P90: {np.percentile(latencies, 90):.0f}ms")
print(f" P99: {np.percentile(latencies, 99):.0f}ms")

# Check budget
budget_ms = 40.0
p99 = np.percentile(latencies, 99)
if p99 < budget_ms:
    print(f"\n✓ P99 {p99:.0f}ms < {budget_ms}ms budget — ON TARGET")
else:
    print(f"\n⚠ P99 {p99:.0f}ms > {budget_ms}ms budget — needs optimization")


# ── Save JSON output ───────────────────────────────────────
import json
json_path = "outputs/day3_detection_output.json"
with open(json_path, 'w') as f:
    json.dump({
        'source_dataset': selected_source,
        'total_images': len(results_summary),
        'total_detections': total_objects,
        'avg_latency_ms': round(np.mean(latencies), 1),
        'p99_latency_ms': round(np.percentile(latencies, 99), 1),
        'results': results_summary
    }, f, indent=2)

print(f"\n✓ JSON saved: {json_path}")
print("\n Day 3 Step 1 Complete!")
print("Next: Run dataset analysis notebook")
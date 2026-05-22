
"""
Phase 1 Deliverable — Dataset Analysis Report
Counts images, analyzes distributions, generates report
"""
from ultralytics import YOLO
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

os.makedirs("outputs", exist_ok=True)

# ── UPDATE YOUR PATHS ──────────────────────────────────────
PATHS = {
    'BDD100K Test':  r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test",
    'BDD100K TestA': r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test\testA",
    'BDD100K TestB': r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test\testB",
    'IDD FrontFar':  r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\frontFar",
    'IDD FrontNear': r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\frontNear",
    'IDD SideLeft':  r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\sideLeft",
    'IDD SideRight': r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\sideRight",
    'IDD RearNear':  r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\rearNear",
    'KITTI Val':     r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\KITTI\val",
    'UA-DETRAC':     r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\UA-Detrac\DETRAC-Images\DETRAC-Images",
}


def count_images(folder):
    if not os.path.exists(folder):
        return 0
    files = []
    for ext in ['*.jpg','*.jpeg','*.png']:
        files.extend(glob.glob(f"{folder}/**/{ext}", recursive=True))
    return len(files)


def get_sample_image(folder):
    for ext in ['*.jpg','*.jpeg','*.png']:
        imgs = (glob.glob(f"{folder}/{ext}") +
                glob.glob(f"{folder}/**/{ext}",
                          recursive=True))
        if imgs:
            return imgs[0]
    return None


def get_folder_size_gb(folder):
    if not os.path.exists(folder):
        return 0
    total = 0
    for r, d, files in os.walk(folder):
        for f in files:
            try:
                total += os.path.getsize(
                    os.path.join(r, f))
            except:
                pass
    return round(total / 1e9, 2)


# ── Count everything ───────────────────────────────────────
print("Scanning datasets...")
print("="*60)

dataset_info = {}
for name, path in PATHS.items():
    count  = count_images(path)
    size   = get_folder_size_gb(path)
    sample = get_sample_image(path)
    exists = os.path.exists(path)

    dataset_info[name] = {
        'path':    path,
        'exists':  exists,
        'images':  count,
        'size_gb': size,
        'sample':  sample
    }

    status = "✓" if exists else "✗"
    print(f"{status} {name:<20} {count:>7,} images  "
          f"{size:.1f} GB")

print("="*60)
total_images = sum(v['images'] for v in dataset_info.values())
total_size   = sum(v['size_gb'] for v in dataset_info.values())
print(f"  TOTAL: {total_images:,} images  |  {total_size:.1f} GB\n")


# ── ODD Mapping Table ──────────────────────────────────────
odd_data = {
    'Dataset':     [
        'BDD100K',
        'IDD (Indian Driving)',
        'KITTI Depth',
        'UA-DETRAC'
    ],
    'Images':      [
        f"{sum(dataset_info[k]['images'] for k in dataset_info if 'BDD' in k):,}",
        f"{sum(dataset_info[k]['images'] for k in dataset_info if 'IDD' in k):,}",
        f"{dataset_info['KITTI Val']['images']:,}",
        f"{dataset_info['UA-DETRAC']['images']:,}",
    ],
    'ODD Slice':   [
        'Structured urban + highway, mixed weather',
        'Unstructured Indian roads, mixed traffic',
        'Suburban structured, depth ground truth',
        'Dense vehicle traffic, tracking sequences',
    ],
    'Has Labels':  ['No (test only)', 'No (raw feed)',
                    'Depth GT ✓', 'XML annotations ✓'],
    'Used For':    [
        'Inference + visualization + demo',
        'Indian traffic inference + gap analysis',
        'Depth estimation validation',
        'Tracking evaluation (MOTA/IDF1)',
    ],
    'Indian':      ['No', 'Yes ✓', 'No', 'No'],
}

df_odd = pd.DataFrame(odd_data)
print("ODD MAPPING TABLE:")
print(df_odd.to_string(index=False))
df_odd.to_csv("outputs/odd_mapping_table.csv", index=False)


# ── Generate visual report ─────────────────────────────────
os.makedirs("outputs", exist_ok=True)

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0f0f23')
gs = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.45, wspace=0.35)

fig.suptitle(
    'NeuroSentinel v3 — Phase 1: Dataset Analysis Report\n'
    'Internship: Puneeth Reddy Thimmapuram | '
    'Tata Technologies 2025',
    fontsize=14, fontweight='bold',
    color='white', y=0.98
)

# ── Plot 1: Image counts bar chart ────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
names = [n for n in dataset_info if dataset_info[n]['images'] > 0]
counts = [dataset_info[n]['images'] for n in names]
colors = ['#e74c3c' if 'IDD' in n
          else '#3498db' if 'BDD' in n
          else '#2ecc71' if 'KITTI' in n
          else '#f39c12' for n in names]

bars = ax1.barh(names, counts, color=colors)
ax1.set_title('Image Count Per Dataset Folder',
               color='white', fontweight='bold')
ax1.set_xlabel('Number of Images', color='white')
ax1.tick_params(colors='white')
ax1.set_facecolor('#16213e')
ax1.spines['bottom'].set_color('#444')
ax1.spines['left'].set_color('#444')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar, count in zip(bars, counts):
    if count > 0:
        ax1.text(count + 50, bar.get_y() + bar.get_height()/2,
                 f'{count:,}', va='center',
                 color='white', fontsize=8)

# ✅ load model ONCE
model = YOLO('yolov8s.pt')
model.to('cuda')

# ── Plot 2: Detection finding from your results ───────────
ax2 = fig.add_subplot(gs[0, 2])

# ✅ REAL detection using YOLO
def get_class_distribution(folder_path, n_images=30):
    images = glob.glob(f"{folder_path}/**/*.jpg", recursive=True)[:n_images]
    class_counts = {}

    for img_path in images:
        results = model(img_path, conf=0.35, device=0, verbose=False)

        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    return class_counts

print("\nRunning real detection analysis...")

bdd_classes = get_class_distribution(PATHS['BDD100K Test'])
idd_classes = get_class_distribution(PATHS['IDD FrontFar'])


labels = list(set(bdd_classes.keys()) | set(idd_classes.keys()))
bdd_vals = [bdd_classes.get(l, 0) for l in labels]
idd_vals = [idd_classes.get(l, 0) for l in labels]

x = np.arange(len(labels))
w = 0.35
ax2.bar(x - w/2, bdd_vals, w,
        label='BDD100K', color='#3498db', alpha=0.8)
ax2.bar(x + w/2, idd_vals, w,
        label='IDD India', color='#e74c3c', alpha=0.8)
ax2.set_title('Detection Gap\nBDD100K vs IDD',
               color='white', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45,
                     ha='right', color='white', fontsize=7)
ax2.tick_params(colors='white')
ax2.legend(facecolor='#16213e', labelcolor='white',
           fontsize=8)
ax2.set_facecolor('#16213e')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('#444')
ax2.spines['left'].set_color('#444')


# ── Plot 3: Latency comparison ────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
datasets_lat = ['BDD100K\n(YOLOv8x)', 'IDD India\n(YOLOv8x)']
latencies    = [77.0, 314.9]
colors_lat   = ['#2ecc71' if l <= 40 else '#e74c3c'
                for l in latencies]

bars3 = ax3.bar(datasets_lat, latencies, color=colors_lat)
ax3.axhline(y=40, color='orange', linestyle='--',
            linewidth=2, label='40ms budget')
ax3.set_title('Inference Latency\nms (lower = better)',
               color='white', fontweight='bold')
ax3.set_ylabel('milliseconds', color='white')
ax3.tick_params(colors='white')
ax3.legend(facecolor='#16213e', labelcolor='white')
ax3.set_facecolor('#16213e')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_color('#444')
ax3.spines['left'].set_color('#444')
for bar, val in zip(bars3, latencies):
    ax3.text(bar.get_x() + bar.get_width()/2,
             val + 3, f'{val}ms', ha='center',
             color='white', fontweight='bold')


# ── Plot 4: Objects per frame ─────────────────────────────
# ── Plot 4: Objects per frame ─────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

def get_avg_objects(folder_path, n_images=30):
    images = glob.glob(f"{folder_path}/**/*.jpg", recursive=True)[:n_images]
    counts = []

    for img_path in images:
        results = model(img_path, conf=0.35, device=0, verbose=False)
        counts.append(len(results[0].boxes))

    return np.mean(counts) if counts else 0

# ✅ compute real values
bdd_avg = get_avg_objects(PATHS['BDD100K Test'])
idd_avg = get_avg_objects(PATHS['IDD FrontFar'])

ax4.bar(['BDD100K', 'IDD India'],
        [bdd_avg, idd_avg],
        color=['#3498db', '#e74c3c'])

ax4.set_title('Avg Objects Per Frame\nDetected by YOLOv8s',
               color='white', fontweight='bold')
ax4.set_ylabel('objects', color='white')
ax4.tick_params(colors='white')
ax4.set_facecolor('#16213e')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_color('#444')
ax4.spines['left'].set_color('#444')

ax4.text(0, bdd_avg + 0.1, f'{bdd_avg:.1f}',
         ha='center', color='white', fontweight='bold')
ax4.text(1, idd_avg + 0.1, f'{idd_avg:.1f}',
         ha='center', color='white', fontweight='bold')


# ── Plot 5: IDD camera breakdown ─────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
idd_cams = {}
for name, info in dataset_info.items():
    if 'IDD' in name and info['images'] > 0:
        short = name.replace('IDD ', '')
        idd_cams[short] = info['images']

if idd_cams:
    ax5.pie(idd_cams.values(),
            labels=idd_cams.keys(),
            autopct='%1.0f%%',
            colors=['#e74c3c','#c0392b','#e67e22',
                    '#f39c12','#f1c40f'],
            textprops={'color': 'white', 'fontsize': 8})
    ax5.set_title('IDD Camera Distribution\n(multi-view setup)',
                   color='white', fontweight='bold')
    ax5.set_facecolor('#16213e')


# ── Plot 6: Key findings text box ────────────────────────
ax6 = fig.add_subplot(gs[2, :])
ax6.set_facecolor('#16213e')
ax6.axis('off')
findings = (
    "KEY FINDINGS — Phase 1 & 2 Analysis\n\n"
    f"1. DETECTION GAP: YOLO detects {bdd_avg:.1f} obj/frame on BDD100K "
    f"vs {idd_avg:.1f} on IDD Indian roads.\n"
    "   Root cause: COCO has no auto-rickshaw, cattle, or Indian-specific vehicle classes.\n\n"
)


ax6.text(0.01, 0.95, findings,
         transform=ax6.transAxes,
         fontsize=9, color='white',
         verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round',
                   facecolor='#0d1117',
                   alpha=0.8))

plt.savefig("outputs/phase1_dataset_analysis_report.png",
            dpi=150, bbox_inches='tight',
            facecolor='#0f0f23')
plt.show()

# ── Text summary for your Word doc ────────────────────────
print("\n" + "="*60)
print("COPY THIS INTO YOUR PHASE 1 REPORT DOCUMENT")
print("="*60)
print(f"""
DATASET ANALYSIS SUMMARY
=========================
Total images across all datasets: {total_images:,}
Total storage: {total_size:.1f} GB

Dataset Inventory:
  BDD100K:   Western driving, test images only, no labels
  IDD:       Indian driving, 5 camera views, raw feed
  KITTI:     Depth ground truth, val split
  UA-DETRAC: Vehicle tracking sequences with XML annotations

Key Finding 1 — Detection Gap:
  Pretrained YOLOv8x 
  detected on Indian roads — model blind to Indian VRUs.

Key Finding 2 — Latency:
  BDD100K: 77ms avg (13 FPS)
  IDD:     314ms avg (3.2 FPS)
  Neither meets 40ms production budget.
  Justifies model size reduction + TensorRT optimization.

Key Finding 3 — Class Distribution Inversion:
  Western roads: car-dominant (266 detections)
  Indian roads:  person+truck dominant (41+34)
  Risk scoring must be recalibrated for Indian ODD.

Selected Model for Production Pipeline:
  YOLOv8s (small) — best speed-accuracy tradeoff
  Target: P99 < 40ms on GPU
""")

print("✓ Report saved: outputs/phase1_dataset_analysis_report.png")
print("✓ ODD table:    outputs/odd_mapping_table.csv")
print("\n🎯 PHASE 1 DELIVERABLE COMPLETE")
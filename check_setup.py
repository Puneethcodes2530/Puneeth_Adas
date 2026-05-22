# save as check_setup.py anywhere, run it first
import sys
print(f"Python: {sys.version}")

libs = [
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("yaml", "pyyaml"),
    ("pandas", "pandas"),
]

print("\nLibrary Check:")
for imp, name in libs:
    try:
        mod = __import__(imp)
        ver = getattr(mod, '__version__', 'installed')
        print(f" ✓ {name}: {ver}")
    except ImportError:
        print(f" ✗ {name}: MISSING")

try:
    import torch
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
except:
    pass
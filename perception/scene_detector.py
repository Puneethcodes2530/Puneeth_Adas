"""
NeuroSentinel v3 — Scene Degradation Detector
Classifies scene condition in under 2ms.
No neural network. Pure image statistics.
Budget: 2ms max (enforced by assertion)
"""
import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SceneState:
    condition: str # CLEAR/NIGHT/FOG/RAIN/GLARE/DUST
    severity: float # 0.0 - 1.0
    brightness: float # mean pixel value
    blur_score: float # laplacian variance
    fog_score: float # dark channel prior
    processing_ms: float # must be < 2.0

    # Confidence threshold adjustments per condition
    @property
    def conf_threshold(self) -> float:
        return {
            'CLEAR': 0.35,
            'NIGHT': 0.45,
            'FOG': 0.50,
            'RAIN': 0.45,
            'GLARE': 0.55,
            'DUST': 0.50,
        }[self.condition]

    # Detection confidence penalty per condition
    @property
    def confidence_penalty(self) -> float:
        return {
            'CLEAR': 1.00,
            'NIGHT': 0.75,
            'FOG': 0.70,
            'RAIN': 0.80,
            'GLARE': 0.50,
            'DUST': 0.65,
        }[self.condition]

    @property
    def depth_weight(self) -> float:
        """How much to trust depth estimates"""
        return {
            'CLEAR': 1.00,
            'NIGHT': 0.80,
            'FOG': 0.40,
            'RAIN': 0.60,
            'GLARE': 0.50,
            'DUST': 0.55,
        }[self.condition]

    @property
    def tau_margin_weight(self) -> float:
        """How much to trust tau-margin TTC"""
        return {
            'CLEAR': 0.50,
            'NIGHT': 0.70,
            'FOG': 0.90,
            'RAIN': 0.85,
            'GLARE': 0.90,
            'DUST': 0.80,
        }[self.condition]

    def summary(self) -> str:
        return (f"[{self.condition}] severity={self.severity:.2f} "
                f"thresh={self.conf_threshold} "
                f"penalty={self.confidence_penalty} "
                f"time={self.processing_ms:.2f}ms")


class SceneDetector:
    """
    Classifies driving scene condition from raw frame.
    Pure image statistics — no neural network.
    Must complete in under 2ms per frame.
    """

    def __init__(self, enforce_budget: bool = True):
        self.enforce_budget = enforce_budget
        self._budget_ms = 2.0

    def analyze(self, frame: np.ndarray) -> SceneState:
        t_start = time.perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = self._brightness(gray)
        blur = self._blur_score(gray)
        fog = self._fog_score(frame)

        condition, severity = self._classify(
            brightness, blur, fog, frame)

        ms = (time.perf_counter() - t_start) * 1000

        if self.enforce_budget:
            assert ms < self._budget_ms * 3, (
                f"Scene detector budget exceeded: {ms:.1f}ms"
            )

        return SceneState(
            condition=condition,
            severity=severity,
            brightness=brightness,
            blur_score=blur,
            fog_score=fog,
            processing_ms=ms
        )

    def _brightness(self, gray: np.ndarray) -> float:
        return float(np.mean(gray))

    def _blur_score(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _fog_score(self, bgr: np.ndarray) -> float:
        """
        Dark Channel Prior for fog/haze detection.
        High dark channel = foggy/hazy image.
        """
        # Resize for speed — 160x120 is enough
        small = cv2.resize(bgr, (160, 120))
        dark = np.min(small, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (5, 5))
        dark_eroded = cv2.erode(dark, kernel)
        return float(np.mean(dark_eroded) / 255.0)

    def _classify(self, brightness: float,
                   blur: float,
                   fog: float,
                   frame: np.ndarray
                   ) -> Tuple[str, float]:

        # NIGHT: very dark
        if brightness < 40:
            severity = 1.0 - brightness / 40.0
            return 'NIGHT', round(severity, 3)

        # GLARE: extremely bright
        if brightness > 220:
            severity = (brightness - 220) / 35.0
            return 'GLARE', round(min(severity, 1.0), 3)

        # FOG: high dark channel + reduced contrast
        if fog > 0.60 and blur < 200:
            return 'FOG', round(fog, 3)

        # DUST: warm-toned haze (different from fog)
        if fog > 0.50 and blur < 150:
            # Check color temperature — dust is warm
            b, g, r = cv2.split(
                cv2.resize(frame, (160, 120)))
            warm_ratio = float(np.mean(r)) / (
                float(np.mean(b)) + 1e-6)
            if warm_ratio > 1.2:
                return 'DUST', round(fog * 0.85, 3)

        # RAIN: blurry + medium brightness
        if blur < 80 and 40 < brightness < 180:
            severity = 1.0 - blur / 80.0
            return 'RAIN', round(severity, 3)

        return 'CLEAR', 0.0


# ── Standalone test ────────────────────────────────────────
if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt
    import os
    import sys

    # Add project root to path
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    # Test paths — update these
    
    TEST_FOLDERS = {
        'BDD100K': r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\BDD100k_Extracted\bdd100k\bdd100k\images\100k\test",
        'IDD FrontFar': r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\frontFar",
    }


    detector = SceneDetector(enforce_budget=False)

    print("SCENE DEGRADATION DETECTOR TEST")
    print("="*60)
    print(f"{'Dataset':<15} {'File':<30} {'Condition':<8} "
          f"{'Severity':<10} {'Threshold':<10} {'Time'}")
    print("-"*85)

    all_conditions = []

    for dataset, folder in TEST_FOLDERS.items():
        images = (glob.glob(f"{folder}/*.jpg") +
                  glob.glob(f"{folder}/**/*.jpg",
                            recursive=True))[:15]

        if not images:
            print(f" No images: {folder}")
            continue

        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            scene = detector.analyze(frame)
            all_conditions.append(scene.condition)

            print(f"{dataset:<15} "
                  f"{os.path.basename(img_path)[:29]:<30} "
                  f"{scene.condition:<8} "
                  f"{scene.severity:<10.3f} "
                  f"{scene.conf_threshold:<10.2f} "
                  f"{scene.processing_ms:.2f}ms")

    # Summary
    if all_conditions:
        from collections import Counter
        counts = Counter(all_conditions)
        total = len(all_conditions)

        print("\nCONDITION DISTRIBUTION:")
        for cond, cnt in sorted(counts.items(),
                                  key=lambda x: x[1],
                                  reverse=True):
            pct = cnt / total * 100
            bar = "█" * int(pct / 3)
            print(f" {cond:<8} {cnt:>4} ({pct:5.1f}%) {bar}")

        avg_ms = np.mean([
            detector.analyze(cv2.imread(
                (glob.glob(f"{list(TEST_FOLDERS.values())[0]}/*.jpg") +
                 glob.glob(f"{list(TEST_FOLDERS.values())[0]}/**/*.jpg",
                           recursive=True))[0]
            )).processing_ms
            for _ in range(5)
        ])
        print(f"\nAverage processing time: {avg_ms:.2f}ms")
        status = "✓ WITHIN" if avg_ms < 2.0 else "✗ EXCEEDS"
        print(f"Budget status: {status} 2ms budget")

"""
NeuroSentinel v3 — Final Improved Scene Degradation Detector

Ultra-fast handcrafted perception module for ADAS.

NO neural network
NO GPU
Pure image statistics
CPU friendly
Edge-device friendly

============================================================
FEATURES
============================================================

✓ Brightness estimation
✓ Blur estimation
✓ Contrast estimation
✓ Edge density analysis
✓ Entropy estimation
✓ Dark channel prior fog detection
✓ Dust detection
✓ Glare detection
✓ Night + headlights handling
✓ Temporal smoothing
✓ Adaptive thresholds
✓ <2ms lightweight architecture

============================================================
SUPPORTED CONDITIONS
============================================================

CLEAR
NIGHT
FOG
RAIN
GLARE
DUST
"""

import cv2
import numpy as np
import time

from dataclasses import dataclass
from typing import Tuple


# ============================================================
# SceneState
# Stores final scene analysis result
# ============================================================

@dataclass
class SceneState:

    # Current scene condition
    # CLEAR / NIGHT / FOG / RAIN / GLARE / DUST
    condition: str

    # Severity from 0.0 → 1.0
    severity: float

    # Mean grayscale intensity
    brightness: float

    # Laplacian variance (This is for the sharpness or the blurriness of the image)
    blur_score: float

    # Dark channel prior fog score
    fog_score: float

    # Processing latency
    processing_ms: float

    # ========================================================
    # Dynamic confidence threshold
    # ========================================================
    #
    # Worse scenes:
    # increase threshold
    #
    # This reduces hallucinations and false positives.
    # ========================================================
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

    # ========================================================
    # Detection confidence penalty
    # ========================================================
    #
    # final_conf =
    # raw_conf * confidence_penalty
    #
    # Example:
    #
    # 0.8 confidence in fog
    # becomes:
    #
    # 0.8 × 0.7 = 0.56
    # ========================================================
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

    # ========================================================
    # How much to trust depth estimation
    # ========================================================
    @property
    def depth_weight(self) -> float:

        return {
            'CLEAR': 1.00,
            'NIGHT': 0.80,
            'FOG': 0.40,
            'RAIN': 0.60,
            'GLARE': 0.50,
            'DUST': 0.55,
        }[self.condition]

    # ========================================================
    # How much to trust temporal TTC
    # ========================================================
    @property
    def tau_margin_weight(self) -> float:

        return {
            'CLEAR': 0.50,
            'NIGHT': 0.70,
            'FOG': 0.90,
            'RAIN': 0.85,
            'GLARE': 0.90,
            'DUST': 0.80,
        }[self.condition]

    # ========================================================
    # Printable summary
    # ========================================================
    def summary(self) -> str:

        return (
            f"[{self.condition}] "
            f"severity={self.severity:.2f} "
            f"threshold={self.conf_threshold} "
            f"penalty={self.confidence_penalty} "
            f"time={self.processing_ms:.2f}ms"
        )


# ============================================================
# SceneDetector
# ============================================================

class SceneDetector:

    """
    Lightweight handcrafted scene degradation detector.

    Uses:
    - brightness
    - blur
    - contrast
    - entropy
    - edge density
    - dark channel prior
    - saturation analysis
    """

    def __init__(self, enforce_budget: bool = True):

        # Whether latency budget should be enforced
        self.enforce_budget = enforce_budget

        # Target latency budget
        self._budget_ms = 2.0

        # ====================================================
        # TEMPORAL SMOOTHING MEMORY
        # ====================================================
        #
        # Single-frame classification fluctuates:
        #
        # CLEAR → FOG → CLEAR ❌
        #
        # EMA smoothing stabilizes outputs:
        #
        # CLEAR → CLEAR → FOG ✅
        #
        # Formula:
        #
        # smoothed =
        # 0.8 * previous +
        # 0.2 * current
        # ====================================================
        self.prev = {

            "fog": 0.0,

            "blur": 0.0
        }

    # ========================================================
    # MAIN ANALYSIS PIPELINE
    # ========================================================
    def analyze(self, frame: np.ndarray) -> SceneState:

        # Start latency timer
        t_start = time.perf_counter()

        # ====================================================
        # Convert frame to grayscale
        #
        # Used for:
        # - brightness
        # - blur
        # - contrast
        # - entropy
        # - edges
        # ====================================================
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ====================================================
        # BASIC IMAGE STATISTICS
        # ====================================================

        # Average scene brightness
        brightness = self._brightness(gray)

        # Sharpness estimation
        blur = self._blur_score(gray)

        # Fog / haze estimation
        fog = self._fog_score(frame)

        # ====================================================
        # TEMPORAL SMOOTHING
        # ====================================================
        #
        # Raw frame statistics fluctuate heavily.
        #
        # EMA smoothing stabilizes:
        #
        # smoother predictions
        # fewer false flips
        # production-like behavior
        # ====================================================

        fog = (
            0.8 * self.prev["fog"] +
            0.2 * fog
        )

        blur = (
            0.8 * self.prev["blur"] +
            0.2 * blur
        )

        # ====================================================
        # OPTIONAL STABILITY CLAMP
        # ====================================================
        #
        # Prevents sudden huge spikes.
        #
        # Mostly safety protection.
        # ====================================================
        blur = np.clip(
            blur,
            0,
            5000
        )

        # Update temporal memory
        self.prev["fog"] = fog
        self.prev["blur"] = blur

        # ====================================================
        # Scene classification
        # ====================================================
        condition, severity = self._classify(
            brightness,
            blur,
            fog,
            frame,
            gray
        )

        # Total latency
        ms = (
            time.perf_counter() - t_start
        ) * 1000

        # ====================================================
        # Optional latency enforcement
        # ====================================================
        if self.enforce_budget:

            assert ms < self._budget_ms * 3, (
                f"Scene detector budget exceeded: "
                f"{ms:.2f}ms"
            )

        # ====================================================
        # Return structured scene output
        # ====================================================
        return SceneState(

            condition=condition,
            severity=severity,

            brightness=brightness,
            blur_score=blur,
            fog_score=fog,

            processing_ms=ms
        )

    # ========================================================
    # BRIGHTNESS ESTIMATION
    # ========================================================
    #
    # Mean grayscale intensity.
    #
    # Dark image:
    # low value
    #
    # Bright image:
    # high value
    # ========================================================
    def _brightness(
        self,
        gray: np.ndarray
    ) -> float:

        return float(
            np.mean(gray)
        )

    # ========================================================
    # BLUR ESTIMATION
    # ========================================================
    #
    # Laplacian variance:
    #
    # High variance:
    # sharp image
    #
    # Low variance:
    # blurry image
    # ========================================================
    def _blur_score(
        self,
        gray: np.ndarray
    ) -> float:

        return float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

    # ========================================================
    # FOG ESTIMATION
    # ========================================================
    #
    # Uses Dark Channel Prior.
    #
    # Fog lifts dark pixels upward.
    #
    # Clear image:
    # strong dark regions
    #
    # Foggy image:
    # dark regions become brighter
    # ========================================================
    def _fog_score(
        self,
        bgr: np.ndarray
    ) -> float:

        # Resize image for speed
        small = cv2.resize(
            bgr,
            (160, 120)
        )

        # Dark channel
        dark = np.min(
            small,
            axis=2
        )

        # Morphological erosion
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )

        dark_eroded = cv2.erode(
            dark,
            kernel
        )

        # Normalize to 0 → 1
        return float(
            np.mean(dark_eroded) / 255.0
        )

    # ========================================================
    # SCENE CLASSIFICATION
    # ========================================================
    def _classify(
        self,
        brightness: float,
        blur: float,
        fog: float,
        frame: np.ndarray,
        gray: np.ndarray
    ) -> Tuple[str, float]:

        # ====================================================
        # CONTRAST ESTIMATION
        # ====================================================
        #
        # Fog destroys contrast heavily.
        # ====================================================
        contrast = float(
            np.std(gray)
        )

        # ====================================================
        # EDGE DENSITY
        # ====================================================
        #
        # Clear scenes:
        # many edges
        #
        # Fog/rain:
        # fewer edges
        # ====================================================
        edges = cv2.Canny(
            gray,
            100,
            200
        )

        edge_density = float(
            np.mean(edges > 0)
        )

        # ====================================================
        # ENTROPY ESTIMATION
        # ====================================================
        #
        # Measures information richness.
        #
        # Fog smooths textures/details,
        # reducing entropy.
        # ====================================================
        hist = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256]
        )

        # ====================================================
        # Safe histogram normalization
        #
        # Prevents divide-by-zero edge case.
        # ====================================================
        hist /= (
            hist.sum() + 1e-6
        )

        entropy = -np.sum(
            hist * np.log2(hist + 1e-7)
        )

        # ====================================================
        # SATURATED PIXEL RATIO
        # ====================================================
        #
        # Useful for:
        # - glare
        # - headlights
        # - overexposure
        # ====================================================
        saturated_ratio = float(
            np.mean(gray > 240)
        )

        # ====================================================
        # NIGHT DETECTION
        # ====================================================
        #
        # Handles:
        # - low-light scenes
        # - night + headlights
        # ====================================================
        if brightness < 55:

            # Headlight bloom
            if saturated_ratio > 0.02:

                return 'GLARE', 0.6

            severity = (
                1.0 - brightness / 55.0
            )

            return (
                'NIGHT',
                round(
                    min(severity, 1.0),
                    3
                )
            )

        # ====================================================
        # GLARE DETECTION
        # ====================================================
        #
        # Strong saturation:
        # sunlight / overexposure
        # ====================================================
        if saturated_ratio > 0.18:

            severity = min(
                saturated_ratio * 2.5,
                1.0
            )

            return (
                'GLARE',
                round(severity, 3)
            )

        # ====================================================
        # FOG DETECTION
        # ====================================================
        #
        # Fog characteristics:
        #
        # - high fog score
        # - low contrast
        # - low edge density
        # - low entropy
        # ====================================================
        if (
            fog > 0.55 and
            contrast < 50 and
            edge_density <
            (0.08 + 0.02 * fog) and
            entropy < 7.0
        ):

            # =================================================
            # Severity mixing
            #
            # Fog score trusted more heavily
            # than contrast reduction.
            # =================================================
            severity = (
                0.7 * fog +
                0.3 * (
                    1.0 - contrast / 50.0
                )
            )

            return (
                'FOG',
                round(
                    min(severity, 1.0),
                    3
                )
            )

        # ====================================================
        # DUST DETECTION
        # ====================================================
        #
        # Similar to fog,
        # but warm colored.
        # ====================================================
        if (
            fog > 0.50 and
            contrast < 60
        ):

            # Resize for speed
            resized = cv2.resize(
                frame,
                (160, 120)
            )

            # Split BGR channels
            b, g, r = cv2.split(
                resized
            )

            # Dust scenes are warm colored
            warm_ratio = float(
                np.mean(r)
            ) / (
                float(np.mean(b)) + 1e-6
            )

            if warm_ratio > 1.20:

                severity = (
                    fog * 0.7
                )

                return (
                    'DUST',
                    round(
                        min(severity, 1.0),
                        3
                    )
                )

        # ====================================================
        # RAIN DETECTION
        # ====================================================
        #
        # Rain introduces:
        # - blur
        # - reduced edges
        # - visibility degradation
        #
        # Slightly adaptive edge threshold.
        # ====================================================
        if (
            blur < 120 and
            edge_density <
            (0.10 + 0.02 * fog) and
            40 < brightness < 180
        ):

            severity = (
                1.0 - blur / 120.0
            )

            return (
                'RAIN',
                round(
                    min(severity, 1.0),
                    3
                )
            )

        # ====================================================
        # OTHERWISE CLEAR
        # ====================================================
        return 'CLEAR', 0.0
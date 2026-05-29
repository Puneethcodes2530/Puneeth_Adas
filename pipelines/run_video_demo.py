"""
NeuroSentinel v3 — Video Demo Runner

Purpose:
- Read an input driving video
- Run NeuroSentinel Phase 4 pipeline
- Perform object detection, tracking, distance estimation, TTC and risk scoring
- Save annotated ADAS demo video
- Save keyframes
- Save JSON summary

How to run:
python pipelines/run_video_demo.py
"""

import os
import sys
import cv2
import time
import json
import numpy as np


# ============================================================
# USER CONFIG — CHANGE ONLY THESE IF NEEDED
# ============================================================

VIDEO_IN = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\videos\idd_hyd_demo.mp4"

VIDEO_OUT = r"outputs\videos\neurosentinel_video_demo.mp4"

MAX_FRAMES = 150

SAVE_KEYFRAME_EVERY = 30

# ============================================================
# DEMO MODE SETTINGS
# ============================================================
#
# For better visual demo:
# - yolov8x gives stronger detection
# - imgsz 960 helps far/small objects
# - conf 0.18 allows more detections
# - depth OFF first keeps demo smoother
# ============================================================

MODEL_WEIGHTS = "yolov8x.pt"
IMG_SIZE = 960
CONF_THRESHOLD = 0.18

USE_DEPTH = True

SHOW_LIVE = False


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = os.path.abspath(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

sys.path.insert(
    0,
    ROOT
)

print("Project root:", ROOT)


# ============================================================
# IMPORTS
# ============================================================

from perception.tracker import Phase4TrackerPipeline

if USE_DEPTH:
    from perception.depth_estimator import DepthEstimatorDA


# ============================================================
# OUTPUT PATHS
# ============================================================

VIDEO_OUT = os.path.join(
    ROOT,
    VIDEO_OUT
)

VIDEO_OUT_DIR = os.path.dirname(
    VIDEO_OUT
)

KEYFRAME_DIR = os.path.join(
    ROOT,
    "outputs",
    "video_keyframes"
)

SUMMARY_PATH = os.path.join(
    ROOT,
    "outputs",
    "video_demo_summary.json"
)

os.makedirs(
    VIDEO_OUT_DIR,
    exist_ok=True
)

os.makedirs(
    KEYFRAME_DIR,
    exist_ok=True
)


# ============================================================
# DEPTH ENGINE
# ============================================================

depth_engine = None

if USE_DEPTH:

    print("[INFO] Loading Depth Engine...")

    depth_engine = DepthEstimatorDA()

else:

    print("[INFO] Depth disabled for demo detection run.")


def depth_fn(frame):
    """
    Function passed into tracker pipeline.
    Returns depth map or None.
    """

    if depth_engine is None:
        return None

    depth_output = depth_engine.estimate(
        frame
    )

    return depth_output.depth_map


# ============================================================
# RISK HELPERS
# ============================================================

RISK_ORDER = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4
}


def get_top_risk_object(objects):
    """
    Returns highest-risk object from tracked objects.
    """

    if len(objects) == 0:
        return None

    return sorted(
        objects,
        key=lambda obj: (
            RISK_ORDER.get(obj.risk, 0),
            obj.risk_score
        ),
        reverse=True
    )[0]


# ============================================================
# EXTRA VIDEO HUD
# ============================================================

def draw_extra_hud(
    vis,
    frame_idx,
    source_fps,
    objects,
    latency_ms
):
    """
    Adds bottom HUD summarizing top-risk object.
    """

    h, w = vis.shape[:2]

    top_obj = get_top_risk_object(
        objects
    )

    if top_obj is None:

        top_class = "None"
        top_risk = "LOW"
        top_distance = "N/A"
        top_ttc = "N/A"
        risk_score = 0.0

    else:

        top_class = top_obj.class_name
        top_risk = top_obj.risk
        top_distance = f"{top_obj.distance_m:.1f}m"
        risk_score = top_obj.risk_score

        if top_obj.ttc.value is None:
            top_ttc = "N/A"
        else:
            top_ttc = f"{top_obj.ttc.value:.2f}s"

    # --------------------------------------------------------
    # Bottom panel
    # --------------------------------------------------------

    cv2.rectangle(
        vis,
        (0, h - 92),
        (w, h),
        (0, 0, 0),
        -1
    )

    line1 = (
        f"Frame:{frame_idx} | "
        f"Time:{frame_idx / max(source_fps, 1):.1f}s | "
        f"Objects:{len(objects)} | "
        f"Latency:{latency_ms:.0f}ms"
    )

    line2 = (
        f"Top Risk:{top_risk} | "
        f"Object:{top_class} | "
        f"Distance:{top_distance} | "
        f"TTC:{top_ttc} | "
        f"Score:{risk_score:.2f}"
    )

    cv2.putText(
        vis,
        line1,
        (12, h - 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        vis,
        line2,
        (12, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 200),
        2,
        cv2.LINE_AA
    )

    return vis


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NeuroSentinel v3 — Video Demo")
    print("=" * 70)

    print("Input video :", VIDEO_IN)
    print("Output video:", VIDEO_OUT)
    print("Model       :", MODEL_WEIGHTS)
    print("Image size  :", IMG_SIZE)
    print("Confidence  :", CONF_THRESHOLD)
    print("Use depth   :", USE_DEPTH)

    if not os.path.exists(VIDEO_IN):

        print("\n[ERROR] Input video not found.")
        print("Check VIDEO_IN path at top of this file.")
        return

    cap = cv2.VideoCapture(
        VIDEO_IN
    )

    if not cap.isOpened():

        print("\n[ERROR] Could not open input video.")
        return

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if source_fps is None or source_fps <= 1:
        source_fps = 10.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f"Video info  : {width}x{height} "
        f"@ {source_fps:.1f} FPS | "
        f"frames={total_frames}"
    )

    # --------------------------------------------------------
    # Load pipeline
    # --------------------------------------------------------

    print("\n[INFO] Loading Phase 4 pipeline...")

    pipe = Phase4TrackerPipeline(
        model_weights=MODEL_WEIGHTS,
        depth_fn=depth_fn if USE_DEPTH else None,
        fps=source_fps,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=0.45,
        use_tracker=True
    )

    print("[INFO] Pipeline loaded.")

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    writer = cv2.VideoWriter(
        VIDEO_OUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height)
    )

    if not writer.isOpened():

        print("[ERROR] Could not open VideoWriter.")
        cap.release()
        return

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    frame_idx = 0
    processed = 0

    latencies = []
    object_counts = []

    risk_counts = {
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 0,
        "CRITICAL": 0
    }

    fcw_events = []
    aeb_events = []

    min_ttc_seen = None

    print("\nProcessing video...")
    print("-" * 80)

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        if MAX_FRAMES is not None and processed >= MAX_FRAMES:
            break

        t0 = time.perf_counter()

        objects, depth_map, pipeline_latency = pipe.process(
            frame
        )

        vis = pipe.draw(
            frame,
            objects,
            pipeline_latency
        )

        total_ms = (
            time.perf_counter() -
            t0
        ) * 1000

        vis = draw_extra_hud(
            vis,
            frame_idx,
            source_fps,
            objects,
            total_ms
        )

        writer.write(
            vis
        )

        processed += 1

        latencies.append(
            total_ms
        )

        object_counts.append(
            len(objects)
        )

        # ----------------------------------------------------
        # Event logic
        # ----------------------------------------------------

        top_obj = get_top_risk_object(
            objects
        )

        top_risk = "LOW"
        top_ttc = None

        if top_obj is not None:

            top_risk = top_obj.risk
            top_ttc = top_obj.ttc.value

            risk_counts[top_risk] = (
                risk_counts.get(top_risk, 0) + 1
            )

            if top_ttc is not None:

                if min_ttc_seen is None:
                    min_ttc_seen = top_ttc
                else:
                    min_ttc_seen = min(
                        min_ttc_seen,
                        top_ttc
                    )

            if top_risk in ["HIGH", "CRITICAL"]:

                fcw_events.append(
                    {
                        "frame": frame_idx,
                        "time_s": round(frame_idx / source_fps, 2),
                        "risk": top_risk,
                        "class": top_obj.class_name,
                        "distance_m": top_obj.distance_m,
                        "ttc_s": top_ttc,
                        "risk_score": top_obj.risk_score
                    }
                )

            if top_risk == "CRITICAL":

                aeb_events.append(
                    {
                        "frame": frame_idx,
                        "time_s": round(frame_idx / source_fps, 2),
                        "class": top_obj.class_name,
                        "distance_m": top_obj.distance_m,
                        "ttc_s": top_ttc,
                        "risk_score": top_obj.risk_score
                    }
                )

        # ----------------------------------------------------
        # Save keyframes
        # ----------------------------------------------------

        save_keyframe = False

        if processed % SAVE_KEYFRAME_EVERY == 0:
            save_keyframe = True

        if top_risk in ["HIGH", "CRITICAL"]:
            save_keyframe = True

        if save_keyframe:

            key_path = os.path.join(
                KEYFRAME_DIR,
                f"frame_{frame_idx:05d}_{top_risk}.jpg"
            )

            cv2.imwrite(
                key_path,
                vis
            )

        # ----------------------------------------------------
        # Console progress
        # ----------------------------------------------------

        if processed % 10 == 0:

            if top_ttc is None:
                ttc_text = "N/A"
            else:
                ttc_text = f"{top_ttc:.2f}s"

            print(
                f"Frame:{frame_idx:<6} "
                f"Objects:{len(objects):<3} "
                f"TopRisk:{top_risk:<9} "
                f"TTC:{ttc_text:<8} "
                f"Latency:{total_ms:.0f}ms"
            )

        if SHOW_LIVE:

            cv2.imshow(
                "NeuroSentinel v3 Video Demo",
                vis
            )

            if cv2.waitKey(1) & 0xFF == 27:
                break

    # --------------------------------------------------------
    # Release
    # --------------------------------------------------------

    cap.release()
    writer.release()

    if SHOW_LIVE:
        cv2.destroyAllWindows()

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("VIDEO DEMO COMPLETE")
    print("=" * 70)

    if len(latencies) == 0:

        print("[ERROR] No frames processed.")
        return

    lat_arr = np.array(
        latencies
    )

    avg_latency = float(
        np.mean(lat_arr)
    )

    actual_fps = float(
        1000.0 / avg_latency
    )

    avg_objects = float(
        np.mean(object_counts)
    )

    summary = {
        "video_in": VIDEO_IN,
        "video_out": VIDEO_OUT,
        "frames_processed": int(processed),
        "source_fps": float(source_fps),
        "processed_duration_s": round(processed / source_fps, 2),
        "avg_latency_ms": round(avg_latency, 1),
        "p50_latency_ms": round(float(np.percentile(lat_arr, 50)), 1),
        "p90_latency_ms": round(float(np.percentile(lat_arr, 90)), 1),
        "p99_latency_ms": round(float(np.percentile(lat_arr, 99)), 1),
        "actual_processing_fps": round(actual_fps, 2),
        "avg_objects_per_frame": round(avg_objects, 2),
        "risk_counts": risk_counts,
        "fcw_events": len(fcw_events),
        "aeb_events": len(aeb_events),
        "min_ttc_seen": min_ttc_seen,
        "fcw_timeline_first_10": fcw_events[:10],
        "aeb_timeline_first_10": aeb_events[:10],
        "settings": {
            "model_weights": MODEL_WEIGHTS,
            "img_size": IMG_SIZE,
            "conf_threshold": CONF_THRESHOLD,
            "use_depth": USE_DEPTH,
            "max_frames": MAX_FRAMES
        },
        "note": (
            "TTC may be N/A during early frames because TTC requires "
            "the same tracked object across multiple frames."
        )
    }

    with open(
        SUMMARY_PATH,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(f"Frames processed      : {processed}")
    print(f"Processed duration    : {processed / source_fps:.1f}s")
    print(f"Avg latency           : {avg_latency:.1f}ms")
    print(f"P50 latency           : {np.percentile(lat_arr, 50):.1f}ms")
    print(f"P90 latency           : {np.percentile(lat_arr, 90):.1f}ms")
    print(f"P99 latency           : {np.percentile(lat_arr, 99):.1f}ms")
    print(f"Actual processing FPS : {actual_fps:.2f}")
    print(f"Avg objects/frame     : {avg_objects:.2f}")
    print(f"FCW events            : {len(fcw_events)}")
    print(f"AEB candidates        : {len(aeb_events)}")
    print(f"Min TTC seen          : {min_ttc_seen}")

    print("\nSaved:")
    print("Video   :", VIDEO_OUT)
    print("Summary :", SUMMARY_PATH)
    print("Frames  :", KEYFRAME_DIR)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()

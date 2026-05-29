"""
NeuroSentinel v3 — Make Video From IDD Frame Sequence

Purpose:
- Convert one IDD highquality_16k sequence folder into an MP4 video.
- Use this generated video for the ADAS video demo.

Run:
python pipelines/make_video_from_frames.py
"""

import os
import glob
import cv2


# ============================================================
# CHANGE THIS PATH ONLY
# ============================================================

IMAGE_FOLDER = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\Datasets\IDD\22Gb IDD Detection(Main)\JPEGImages\highquality_16k\HYD-2018-08-24_13-42-50"

VIDEO_OUT = r"C:\Users\PTT933267\Downloads\Puneeth_Adas\videos\idd_hyd_demo.mp4"

FPS = 2
MAX_FRAMES = 300


def find_images(folder):

    patterns = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.png"
    ]

    images = []

    for pattern in patterns:

        images.extend(
            glob.glob(
                os.path.join(folder, pattern),
                recursive=True
            )
        )

    return sorted(
        list(set(images))
    )


def main():

    print("=" * 70)
    print("Creating video from IDD frame sequence")
    print("=" * 70)

    print("Image folder:", IMAGE_FOLDER)
    print("Video out:", VIDEO_OUT)

    if not os.path.exists(IMAGE_FOLDER):

        print("[ERROR] Folder not found:")
        print(IMAGE_FOLDER)
        return

    images = find_images(
        IMAGE_FOLDER
    )

    if len(images) == 0:

        print("[ERROR] No images found inside folder.")
        return

    images = images[:MAX_FRAMES]

    os.makedirs(
        os.path.dirname(VIDEO_OUT),
        exist_ok=True
    )

    first = cv2.imread(
        images[0]
    )

    if first is None:

        print("[ERROR] Could not read first image:")
        print(images[0])
        return

    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        VIDEO_OUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h)
    )

    if not writer.isOpened():

        print("[ERROR] Could not create MP4 writer.")
        print("Try changing VIDEO_OUT extension to .avi and fourcc to XVID.")
        return

    print(f"Images found : {len(images)}")
    print(f"Resolution   : {w} x {h}")
    print(f"FPS          : {FPS}")

    written = 0

    for idx, img_path in enumerate(images):

        frame = cv2.imread(
            img_path
        )

        if frame is None:
            continue

        if frame.shape[:2] != (h, w):

            frame = cv2.resize(
                frame,
                (w, h)
            )

        writer.write(
            frame
        )

        written += 1

        if written % 50 == 0:

            print(
                f"{written}/{len(images)} frames written"
            )

    writer.release()

    print("\nDONE ✅")
    print("Frames written:", written)
    print("Video saved:", VIDEO_OUT)


if __name__ == "__main__":

    main()

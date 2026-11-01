# =============================================================
# track_fish.py  —  Track fish across video frames
# =============================================================
# What this file does:
#   1. Takes a video file as input
#   2. YOLOv8 detects fish in each frame
#   3. ByteTrack assigns unique ID to each fish
#   4. Tracks fish across frames (no double counting)
#   5. Counts unique fish seen in entire video
#   6. Saves output video with tracking boxes
# =============================================================

import cv2
import argparse
import os
from ultralytics import YOLO
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────
YOLO_MODEL = r"C:\Users\ravib\ML_Fish_Classifier\models\yolo_fish\weights\best.pt"
OUTPUT_DIR = r"C:\Users\ravib\ML_Fish_Classifier\models\tracking"
CONF       = 0.25

def track_fish(video_path):
    """
    Track fish across video frames using ByteTrack.

    Args:
        video_path: path to input video file
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────
    model = YOLO(YOLO_MODEL)

    # ── Open video ─────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo     : {video_path}")
    print(f"FPS       : {fps}")
    print(f"Size      : {width}x{height}")
    print(f"Frames    : {total}")

    # ── Setup output video ─────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR,
               "tracked_" + os.path.basename(video_path))
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps,
                               (width, height))

    # ── Tracking variables ─────────────────────────────────────
    unique_ids      = set()   # all unique fish IDs seen
    frame_counts    = []      # fish count per frame
    id_colors       = {}      # color per fish ID

    frame_num = 0

    # ── Process each frame ─────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Run YOLOv8 tracking with ByteTrack
        results = model.track(
            source    = frame,
            conf      = CONF,
            tracker   = "bytetrack.yaml",
            persist   = True,   # keep track IDs across frames
            verbose   = False
        )

        result = results[0]

        # Count fish in this frame
        frame_fish = 0

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids   = result.boxes.id.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, fish_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = map(int, box)

                # Track unique IDs
                unique_ids.add(fish_id)
                frame_fish += 1

                # Assign color per ID (consistent across frames)
                if fish_id not in id_colors:
                    import random
                    random.seed(fish_id)
                    id_colors[fish_id] = (
                        random.randint(50, 255),
                        random.randint(50, 255),
                        random.randint(50, 255)
                    )
                color = id_colors[fish_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              color, 2)

                # Draw ID label
                label = f"ID:{fish_id} {conf*100:.0f}%"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1)

        frame_counts.append(frame_fish)

        # Draw stats on frame
        cv2.putText(frame,
                    f"Frame: {frame_num}/{total}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame,
                    f"Fish in frame: {frame_fish}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame,
                    f"Unique fish total: {len(unique_ids)}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        writer.write(frame)

        # Print progress every 30 frames
        if frame_num % 30 == 0:
            print(f"  Frame {frame_num}/{total} — "
                  f"Fish in frame: {frame_fish} — "
                  f"Unique total: {len(unique_ids)}")

    # ── Cleanup ────────────────────────────────────────────────
    cap.release()
    writer.release()

    # ── Final report ───────────────────────────────────────────
    avg_per_frame = sum(frame_counts) / len(frame_counts) \
                    if frame_counts else 0

    print(f"\n{'='*45}")
    print(f"  FISH TRACKING REPORT")
    print(f"{'='*45}")
    print(f"  Total frames processed : {frame_num}")
    print(f"  Avg fish per frame     : {avg_per_frame:.1f}")
    print(f"  Max fish in one frame  : {max(frame_counts)}")
    print(f"  Unique fish tracked    : {len(unique_ids)}")
    print(f"{'='*45}")
    print(f"\nOutput video → {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True,
                        help="Path to input video file")
    args = parser.parse_args()
    track_fish(args.video)
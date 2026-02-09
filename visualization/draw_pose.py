# visualization/draw_pose.py

import cv2
from .colors import COLORS

POSE_CONNECTIONS = [
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32)
]


def draw_pose(frame, poses, width, height, visibility_thresh=0.5):
    """
    poses: [
        {
            "pose_index": int,
            "landmarks": [{ "x","y","z","visibility" }]
        }
    ]
    """
    for pose in poses:
        lm = pose["landmarks"]

        for a, b in POSE_CONNECTIONS:
            if lm[a]["visibility"] > visibility_thresh and lm[b]["visibility"] > visibility_thresh:
                cv2.line(
                    frame,
                    (int(lm[a]["x"] * width), int(lm[a]["y"] * height)),
                    (int(lm[b]["x"] * width), int(lm[b]["y"] * height)),
                    COLORS["pose"],
                    3
                )

        for p in lm:
            if p["visibility"] > visibility_thresh:
                cv2.circle(
                    frame,
                    (int(p["x"] * width), int(p["y"] * height)),
                    5,
                    COLORS["pose_point"],
                    -1
                )

    return frame

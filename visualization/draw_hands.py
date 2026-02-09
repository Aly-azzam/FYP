# visualization/draw_hands.py

import cv2
from .colors import COLORS

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]


def draw_hands(frame, hands, width, height):
    """
    hands: [
        {
            "hand_index": int,
            "landmarks": [{ "x": float, "y": float, "z": float }]
        }
    ]
    """
    for hand in hands:
        lm = hand["landmarks"]

        for a, b in HAND_CONNECTIONS:
            p1, p2 = lm[a], lm[b]
            cv2.line(
                frame,
                (int(p1["x"] * width), int(p1["y"] * height)),
                (int(p2["x"] * width), int(p2["y"] * height)),
                COLORS["hand"],
                2
            )

        for p in lm:
            cv2.circle(
                frame,
                (int(p["x"] * width), int(p["y"] * height)),
                4,
                COLORS["hand"],
                -1
            )

    return frame

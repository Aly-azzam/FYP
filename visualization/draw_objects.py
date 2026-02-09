# visualization/draw_objects.py

import cv2
from .colors import COLORS


def draw_objects(frame, objects):
    """
    objects: [
        {
            "label": str,
            "confidence": float,
            "bbox": [x1,y1,x2,y2]
        }
    ]
    """
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["bbox"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["object"], 2)
        cv2.putText(
            frame,
            f"{obj['label']} {obj['confidence']:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS["object"],
            2
        )

    return frame

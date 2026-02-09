# visualization/draw_faces.py

import cv2
from .colors import COLORS


def draw_faces(frame, faces):
    """
    faces: [
        {
            "bbox": { "x": int, "y": int, "width": int, "height": int },
            "confidence": float
        }
    ]
    """
    for face in faces:
        b = face["bbox"]
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS["face"], 2)
        cv2.putText(
            frame,
            f"Face {face['confidence']:.2f}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS["face"],
            2
        )

    return frame

# models/mediapipe_hands/detector.py

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class MediaPipeHandsDetector:
    """
    Pure MediaPipe Hands detector.
    - Loads the hand landmarker once
    - Runs detection on a single frame
    - Returns raw landmarks (NO drawing, NO logic)
    """

    def __init__(self, model_path: str, max_hands: int = 2):
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, mp_image):
        """
        Run hand detection on a MediaPipe Image.

        Returns:
        [
            {
                "hand_index": int,
                "landmarks": [
                    {"x": float, "y": float, "z": float},
                    ...
                ]
            },
            ...
        ]
        """
        result = self.detector.detect(mp_image)

        hands_output = []

        if not result.hand_landmarks:
            return hands_output

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            landmarks = []

            for lm in hand_landmarks:
                landmarks.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                })

            hands_output.append({
                "hand_index": idx,
                "landmarks": landmarks
            })

        return hands_output

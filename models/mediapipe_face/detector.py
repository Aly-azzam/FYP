# models/mediapipe_face/detector.py

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeFaceDetector:
    """
    Pure MediaPipe Face detector.
    - Loads the face detector once
    - Runs detection on a single frame
    - Returns raw face bounding boxes (NO drawing, NO logic)
    """

    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.FaceDetectorOptions(
            base_options=base_options
        )

        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, mp_image):
        """
        Run face detection on a MediaPipe Image.

        Returns:
        [
            {
                "bbox": {
                    "x": int,
                    "y": int,
                    "width": int,
                    "height": int
                },
                "confidence": float
            },
            ...
        ]
        """
        result = self.detector.detect(mp_image)

        faces_output = []

        if not result.detections:
            return faces_output

        for detection in result.detections:
            bbox = detection.bounding_box

            faces_output.append({
                "bbox": {
                    "x": bbox.origin_x,
                    "y": bbox.origin_y,
                    "width": bbox.width,
                    "height": bbox.height
                },
                "confidence": detection.categories[0].score
            })

        return faces_output

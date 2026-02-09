# models/mediapipe_pose/detector.py

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipePoseDetector:
    """
    Pure MediaPipe Pose detector.
    - Loads the pose landmarker once
    - Runs detection on a single frame
    - Returns raw pose landmarks (NO drawing, NO logic)
    """

    def __init__(self, model_path: str, max_poses: int = 1):
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=max_poses,
            output_segmentation_masks=False
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, mp_image):
        """
        Run pose detection on a MediaPipe Image.

        Returns:
        [
            {
                "pose_index": int,
                "landmarks": [
                    {
                        "x": float,
                        "y": float,
                        "z": float,
                        "visibility": float
                    },
                    ...
                ]
            },
            ...
        ]
        """
        result = self.detector.detect(mp_image)

        poses_output = []

        if not result.pose_landmarks:
            return poses_output

        for idx, pose_landmarks in enumerate(result.pose_landmarks):
            landmarks = []

            for lm in pose_landmarks:
                landmarks.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

            poses_output.append({
                "pose_index": idx,
                "landmarks": landmarks
            })

        return poses_output

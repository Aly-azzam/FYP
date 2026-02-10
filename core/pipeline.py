# core/pipeline.py

import cv2
import mediapipe as mp
import json

from models.yolo.detector import YoloDetector
from models.mediapipe_hands.detector import MediaPipeHandsDetector
from models.mediapipe_pose.detector import MediaPipePoseDetector
from models.mediapipe_face.detector import MediaPipeFaceDetector

from visualization.draw_faces import draw_faces
from visualization.draw_hands import draw_hands
from visualization.draw_pose import draw_pose
from visualization.draw_objects import draw_objects

from postprocessing.filtering import filter_outliers
from postprocessing.smoothing import smooth_moving_average
from postprocessing.interpolation import interpolate_missing


class Pipeline:
    """
    Central perception pipeline.
    Orchestrates models, postprocessing, and visualization.
    """

    def __init__(self, assets_dir: str):
        # ===== Load models once =====
        self.yolo = YoloDetector(
            model_path=f"{assets_dir}/yolov8n.pt",
            conf=0.2
        )

        self.hands = MediaPipeHandsDetector(
            model_path=f"{assets_dir}/hand_landmarker.task"
        )

        self.pose = MediaPipePoseDetector(
            model_path=f"{assets_dir}/pose_landmarker.task"
        )

        self.face = MediaPipeFaceDetector(
            model_path=f"{assets_dir}/blaze_face_short_range.tflite"
        )

    # ==========================================================
    # IMAGE PIPELINE
    # ==========================================================

    def process_image(self, image_path, output_path, enabled=None):
        """
        Process a single image.
        enabled: dict of booleans, e.g. {"face": True, "hands": True, "pose": True, "objects": True}
        """
        if enabled is None:
            enabled = {"face": True, "hands": True, "pose": True, "objects": True}

        frame = cv2.imread(image_path)
        h, w = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        # ===== Detection (only enabled models) =====
        faces = self.face.detect(mp_image) if enabled.get("face") else []
        hands = self.hands.detect(mp_image) if enabled.get("hands") else []
        poses = self.pose.detect(mp_image) if enabled.get("pose") else []
        objects = self.yolo.detect(frame) if enabled.get("objects") else []

        # ===== Visualization =====
        if enabled.get("face"):
            frame = draw_faces(frame, faces)
        if enabled.get("hands"):
            frame = draw_hands(frame, hands, w, h)
        if enabled.get("pose"):
            frame = draw_pose(frame, poses, w, h)
        if enabled.get("objects"):
            frame = draw_objects(frame, objects)

        cv2.imwrite(output_path, frame)

        models_used = [k for k, v in enabled.items() if v]
        summary = {
            "faces": len(faces),
            "hands": len(hands),
            "poses": len(poses),
            "objects": len(objects),
            "models_used": models_used
        }

        return output_path, summary

    # ==========================================================
    # VIDEO PIPELINE
    # ==========================================================

    def process_video(self, video_path, output_path, enabled=None):
        """
        Process a video file.
        enabled: dict of booleans for each model type.
        """
        if enabled is None:
            enabled = {"face": True, "hands": True, "pose": True, "objects": True}

        cap = cv2.VideoCapture(video_path)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"avc1"),
            fps,
            (w, h)
        )

        # ===== Temporal tracking =====
        time_axis = []
        hand_y = []
        pose_y = []
        total_objects = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            time_axis.append(t)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            # ===== Detection (only enabled) =====
            faces = self.face.detect(mp_image) if enabled.get("face") else []
            hands = self.hands.detect(mp_image) if enabled.get("hands") else []
            poses = self.pose.detect(mp_image) if enabled.get("pose") else []
            objects = self.yolo.detect(frame) if enabled.get("objects") else []

            total_objects += len(objects)

            # ===== Extract trajectories (RAW) =====
            hy = None
            if hands:
                hy = hands[0]["landmarks"][8]["y"]  # index fingertip

            py = None
            if poses:
                py = poses[0]["landmarks"][0]["y"]  # nose

            hand_y.append(hy)
            pose_y.append(py)

            # ===== Visualization =====
            if enabled.get("face"):
                frame = draw_faces(frame, faces)
            if enabled.get("hands"):
                frame = draw_hands(frame, hands, w, h)
            if enabled.get("pose"):
                frame = draw_pose(frame, poses, w, h)
            if enabled.get("objects"):
                frame = draw_objects(frame, objects)

            out.write(frame)

        cap.release()
        out.release()

        # ===== Postprocessing =====
        hand_y = interpolate_missing(
            smooth_moving_average(
                filter_outliers(hand_y)
            )
        )

        pose_y = interpolate_missing(
            smooth_moving_average(
                filter_outliers(pose_y)
            )
        )

        graph_data = {
            "t": time_axis,
            "hand_y": hand_y,
            "pose_y": pose_y
        }

        models_used = [k for k, v in enabled.items() if v]
        summary = {
            "frames": len(time_axis),
            "fps": round(fps, 1),
            "resolution": f"{w}x{h}",
            "duration_sec": round(len(time_axis) / fps, 2),
            "total_objects_detected": total_objects,
            "models_used": models_used
        }

        return output_path, json.dumps(graph_data), summary

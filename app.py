from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


yolo_model = YOLO("yolov8n.pt")  # lightweight & fast


UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===== Skeleton Connections =====

# Hand connections (21 landmarks)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# Pose connections (33 landmarks) - MediaPipe Pose
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]

# Colors for visualization (BGR format)
COLORS = {
    'face': (255, 0, 0),       # Blue
    'hand': (0, 255, 0),       # Green
    'pose': (255, 107, 107),   # Coral/salmon (matches UI accent)
    'pose_point': (255, 165, 0) # Orange
}


# ===== Detector Factory Functions =====

def create_hand_detector():
    model_path = os.path.join(BASE_DIR, "hand_landmarker.task")
    base = python.BaseOptions(model_asset_path=model_path)
    opts = vision.HandLandmarkerOptions(base_options=base, num_hands=2)
    return vision.HandLandmarker.create_from_options(opts)


def create_face_detector():
    model_path = os.path.join(BASE_DIR, "blaze_face_short_range.tflite")
    base = python.BaseOptions(model_asset_path=model_path)
    opts = vision.FaceDetectorOptions(base_options=base)
    return vision.FaceDetector.create_from_options(opts)


def create_pose_detector():
    """Create MediaPipe Pose Landmarker detector."""
    model_path = os.path.join(BASE_DIR, "pose_landmarker.task")
    base = python.BaseOptions(model_asset_path=model_path)
    opts = vision.PoseLandmarkerOptions(
        base_options=base,
        output_segmentation_masks=False,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(opts)


def draw_pose_landmarks(image, pose_landmarks, w, h):
    """Draw pose skeleton on image."""
    if not pose_landmarks:
        return image
    
    img = image.copy()
    
    for pose in pose_landmarks:
        # Draw connections (skeleton lines)
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(pose) and end_idx < len(pose):
                start = pose[start_idx]
                end = pose[end_idx]
                
                # Check visibility
                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(img, start_point, end_point, COLORS['pose'], 3)
        
        # Draw keypoints
        for i, landmark in enumerate(pose):
            if landmark.visibility > 0.5:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (x, y), 5, COLORS['pose_point'], -1)
                cv2.circle(img, (x, y), 7, (255, 255, 255), 1)
    
    return img


# ===== Routes =====

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("media")
    if not file:
        return "No file uploaded"

    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    if filename.lower().endswith((".mp4", ".mov", ".avi")):
        return process_video(input_path)
    else:
        return process_image(input_path)


def process_image(path):
    """Process a single image with face, hand, and pose detection."""
    hand_detector = create_hand_detector()
    face_detector = create_face_detector()
    pose_detector = create_pose_detector()

    img = cv2.imread(path)
    h, w = img.shape[:2]

    # Convert to MediaPipe Image format
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )

    # Detection results for JSON
    detection_results = {
        "faces_detected": 0,
        "hands_detected": 0,
        "poses_detected": 0,
        "pose_landmarks": 0
    }

    # ===== Face Detection =====
    face_res = face_detector.detect(mp_img)
    if face_res.detections:
        detection_results["faces_detected"] = len(face_res.detections)
        for d in face_res.detections:
            b = d.bounding_box
            cv2.rectangle(img, (b.origin_x, b.origin_y),
                          (b.origin_x + b.width, b.origin_y + b.height),
                          COLORS['face'], 2)
            cv2.putText(img, "Face", (b.origin_x, b.origin_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['face'], 2)

    # ===== Hand Detection =====
    hand_res = hand_detector.detect(mp_img)
    if hand_res.hand_landmarks:
        detection_results["hands_detected"] = len(hand_res.hand_landmarks)
        for hand in hand_res.hand_landmarks:
            # Draw connections
            for a, b in HAND_CONNECTIONS:
                p1, p2 = hand[a], hand[b]
                cv2.line(img,
                         (int(p1.x * w), int(p1.y * h)),
                         (int(p2.x * w), int(p2.y * h)),
                         COLORS['hand'], 2)
            # Draw keypoints
            for lm in hand:
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, COLORS['hand'], -1)

    # ===== Pose Detection (MediaPipe Pose) =====
    pose_res = pose_detector.detect(mp_img)
    if pose_res.pose_landmarks:
        detection_results["poses_detected"] = len(pose_res.pose_landmarks)
        # Count visible landmarks
        visible_count = sum(
            1 for pose in pose_res.pose_landmarks 
            for lm in pose if lm.visibility > 0.5
        )
        detection_results["pose_landmarks"] = visible_count
        img = draw_pose_landmarks(img, pose_res.pose_landmarks, w, h)

    out_name = "processed_image.png"
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), img)

    return render_template(
        "result.html",
        media_type="image",
        media_file=out_name,
        graph_data=None,
        json_preview=json.dumps(detection_results, indent=2)
    )

def filter_outliers(values, max_jump=0.15):
    """
    Remove sudden unrealistic jumps in normalized positions.
    """
    if not values:
        return values

    filtered = [values[0]]

    for v in values[1:]:
        prev = filtered[-1]
        if v is None or prev is None:
            filtered.append(v)
        elif abs(v - prev) > max_jump:
            filtered.append(prev)  # keep previous value
        else:
            filtered.append(v)

    return filtered

def smooth_moving_average(values, window=5):
    """
    Smooth signal using moving average.
    """
    smoothed = []

    for i in range(len(values)):
        window_vals = [
            v for v in values[max(0, i - window): i + 1]
            if v is not None
        ]

        if window_vals:
            smoothed.append(sum(window_vals) / len(window_vals))
        else:
            smoothed.append(None)

    return smoothed


def interpolate_missing(values):
    """
    Fill short gaps (None values) using linear interpolation.
    """
    interpolated = values[:]

    for i in range(1, len(values) - 1):
        if interpolated[i] is None:
            prev_v = interpolated[i - 1]
            next_v = interpolated[i + 1]
            if prev_v is not None and next_v is not None:
                interpolated[i] = (prev_v + next_v) / 2

    return interpolated


def process_video(path):
    """
    Process video with:
    - MediaPipe: face, hands, pose
    - YOLO: object detection
    """

    # ===== Initialize detectors =====
    hand_detector = create_hand_detector()
    face_detector = create_face_detector()
    pose_detector = create_pose_detector()

    # ===== Video I/O =====
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_name = "processed_video.mp4"
    out_path = os.path.join(OUTPUT_FOLDER, out_name)

    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (w, h)
    )

    # ===== Data tracking =====
    frame_count = 0
    yolo_objects_count = 0

    # ===== Trajectory tracking (NEW) =====
    time_axis = []
    hand_y = []
    pose_y = []

    print(f"[INFO] Processing video: {total_frames} frames @ {fps:.1f} fps")

    # ===== Main loop =====
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_axis.append(t)

        # Convert to MediaPipe image
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        # ================= FACE =================
        face_res = face_detector.detect(mp_img)
        if face_res.detections:
            for d in face_res.detections:
                b = d.bounding_box
                cv2.rectangle(
                    frame,
                    (b.origin_x, b.origin_y),
                    (b.origin_x + b.width, b.origin_y + b.height),
                    COLORS["face"],
                    2
                )

        # ================= HANDS =================
        hand_res = hand_detector.detect(mp_img)

        h_y = None
        if hand_res.hand_landmarks:
            h_y = hand_res.hand_landmarks[0][8].y  # index fingertip

            for hand in hand_res.hand_landmarks:
                for a, b in HAND_CONNECTIONS:
                    p1, p2 = hand[a], hand[b]
                    cv2.line(
                        frame,
                        (int(p1.x * w), int(p1.y * h)),
                        (int(p2.x * w), int(p2.y * h)),
                        COLORS["hand"],
                        2
                    )
                for lm in hand:
                    cv2.circle(
                        frame,
                        (int(lm.x * w), int(lm.y * h)),
                        4,
                        COLORS["hand"],
                        -1
                    )

        hand_y.append(h_y)

        # ================= POSE =================
        pose_res = pose_detector.detect(mp_img)

        p_y = None
        if pose_res.pose_landmarks:
            p_y = pose_res.pose_landmarks[0][0].y  # nose
            frame = draw_pose_landmarks(frame, pose_res.pose_landmarks, w, h)

        pose_y.append(p_y)

        # ================= YOLO OBJECTS =================
        yolo_results = yolo_model(frame, conf=0.2, verbose=False)

        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo_model.names[cls_id]

                yolo_objects_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

        out.write(frame)

    # ===== Cleanup =====
    cap.release()
    out.release()

    print("[INFO] Video processing complete")

    # ===== JSON post-processing (NEW) =====
    hand_y = filter_outliers(hand_y)
    hand_y = smooth_moving_average(hand_y)
    hand_y = interpolate_missing(hand_y)

    pose_y = filter_outliers(pose_y)
    pose_y = smooth_moving_average(pose_y)
    pose_y = interpolate_missing(pose_y)

    graph_data = {
        "t": time_axis,
        "hand_y": hand_y,
        "pose_y": pose_y,
        "post_processing": {
            "filtering": "outlier removal",
            "smoothing": "moving average",
            "interpolation": "linear"
        }
    }

    # ===== JSON summary =====
    summary = {
        "frames_processed": frame_count,
        "fps": round(fps, 1),
        "resolution": f"{w}x{h}",
        "duration_sec": round(frame_count / fps, 2),
        "yolo_objects_detected": yolo_objects_count,
        "models": {
            "mediapipe": ["face", "hands", "pose"],
            "yolo": "yolov8n"
        }
    }

    return render_template(
        "result.html",
        media_type="video",
        media_file=out_name,
        graph_data=json.dumps(graph_data),
        json_preview=json.dumps(summary, indent=2)
    )


@app.route("/outputs/<filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True)

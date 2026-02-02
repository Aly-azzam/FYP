from ultralytics import YOLO
import cv2
import json
import os


def run_yolo_on_video(
    video_path,
    output_video_path,
    output_json_path,
    model_path="yolov8n.pt",
    conf=0.3,
    frame_step=10
):
    """
    YOLO object detection on video.
    Saves annotated video + JSON detections.
    """

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    json_data = {
        "model": os.path.basename(model_path),
        "media_type": "video",
        "source": os.path.basename(video_path),
        "fps": fps,
        "frames": []
    }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_index % frame_step == 0:
            timestamp = frame_index / fps if fps else None

            frame_data = {
                "frame_index": frame_index,
                "timestamp_sec": timestamp,
                "detections": []
            }

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])

                frame_data["detections"].append({
                    "class_id": cls_id,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

            json_data["frames"].append(frame_data)

    cap.release()
    out.release()

    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return output_video_path, output_json_path


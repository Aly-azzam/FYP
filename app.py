from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Hand skeleton connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

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
    hand_detector = create_hand_detector()
    face_detector = create_face_detector()

    img = cv2.imread(path)
    h, w = img.shape[:2]

    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )

    face_res = face_detector.detect(mp_img)
    if face_res.detections:
        for d in face_res.detections:
            b = d.bounding_box
            cv2.rectangle(img, (b.origin_x, b.origin_y),
                          (b.origin_x+b.width, b.origin_y+b.height),
                          (255,0,0), 2)

    hand_res = hand_detector.detect(mp_img)
    if hand_res.hand_landmarks:
        for hand in hand_res.hand_landmarks:
            for a,b in HAND_CONNECTIONS:
                p1,p2 = hand[a], hand[b]
                cv2.line(img,
                         (int(p1.x*w), int(p1.y*h)),
                         (int(p2.x*w), int(p2.y*h)),
                         (0,255,0), 2)

    out_name = "processed_image.png"
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), img)

    return render_template(
        "result.html",
        media_type="image",
        media_file=out_name,
        graph_data=None,
        json_preview=json.dumps({}, indent=2)
    )

def process_video(path):
    hand_detector = create_hand_detector()
    face_detector = create_face_detector()

    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out_name = "processed_video.mp4"
    out_path = os.path.join(OUTPUT_FOLDER, out_name)

    out = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"avc1"),  # H.264
    fps,
    (w, h)
)

    

    times, ys = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        face_res = face_detector.detect(mp_img)
        if face_res.detections:
            for d in face_res.detections:
                b = d.bounding_box
                cv2.rectangle(frame,
                              (b.origin_x, b.origin_y),
                              (b.origin_x+b.width, b.origin_y+b.height),
                              (255,0,0), 2)

        y_val = None
        hand_res = hand_detector.detect(mp_img)
        if hand_res.hand_landmarks:
            hand = hand_res.hand_landmarks[0]
            for a,b in HAND_CONNECTIONS:
                p1,p2 = hand[a], hand[b]
                cv2.line(frame,
                         (int(p1.x*w), int(p1.y*h)),
                         (int(p2.x*w), int(p2.y*h)),
                         (0,255,0), 2)
            y_val = hand[8].y

        times.append(t)
        ys.append(y_val)
        out.write(frame)

    cap.release()
    out.release()

    graph_data = json.dumps({"t": times, "y": ys})

    return render_template(
        "result.html",
        media_type="video",
        media_file=out_name,
        graph_data=graph_data,
        json_preview=json.dumps({"frames": len(times)}, indent=2)
    )

@app.route("/outputs/<filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)

if __name__ == "__main__":
    app.run(debug=True)

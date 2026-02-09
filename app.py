# app.py

from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename

from core.pipeline import Pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

# Put your weights here:
# assets/models/yolov8n.pt
# assets/models/hand_landmarker.task
# assets/models/pose_landmarker.task
# assets/models/blaze_face_short_range.tflite
ASSETS_MODELS_DIR = os.path.join(BASE_DIR, "assets", "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# IMPORTANT:
# Your JS/CSS are inside templates/static, but they are referenced as /static/...
# So we make Flask serve static files from templates/static.
app = Flask(
    __name__,
    template_folder="templates",
    static_folder=os.path.join("templates", "static"),
    static_url_path="/static",
)

pipeline = Pipeline(assets_dir=ASSETS_MODELS_DIR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("media")
    if not file or file.filename.strip() == "":
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    ext = os.path.splitext(filename.lower())[1]
    is_video = ext in {".mp4", ".mov", ".avi", ".mkv"}

    if is_video:
        out_name = "processed_video.mp4"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)

        out_path, graph_data_json, summary = pipeline.process_video(input_path, out_path)

        return render_template(
            "result.html",
            media_type="video",
            media_file=os.path.basename(out_path),
            graph_data=graph_data_json,  # already JSON string
            json_preview=str(summary).replace("'", '"')  # simple preview
        )

    else:
        out_name = "processed_image.png"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)

        out_path, summary = pipeline.process_image(input_path, out_path)

        return render_template(
            "result.html",
            media_type="image",
            media_file=os.path.basename(out_path),
            graph_data=None,
            json_preview=str(summary).replace("'", '"')
        )


@app.route("/outputs/<filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True)

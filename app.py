# app.py

from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify
import os
import json
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

from core.pipeline import Pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")
ASSETS_MODELS_DIR = os.path.join(BASE_DIR, "assets", "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder=os.path.join("templates", "static"),
    static_url_path="/static",
)

pipeline = Pipeline(assets_dir=ASSETS_MODELS_DIR)


# ===== History Helpers =====

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def add_to_history(entry):
    history = load_history()
    history.insert(0, entry)
    save_history(history)


def get_history_entry(entry_id):
    for entry in load_history():
        if entry.get('id') == entry_id:
            return entry
    return None


# ===== Routes =====

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

    # Read overlay toggles from form
    enabled = {
        "face": request.form.get("toggle_face") == "on",
        "hands": request.form.get("toggle_hands") == "on",
        "pose": request.form.get("toggle_pose") == "on",
        "objects": request.form.get("toggle_objects") == "on",
    }

    ext = os.path.splitext(filename.lower())[1]
    is_video = ext in {".mp4", ".mov", ".avi", ".mkv"}

    # Unique output filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]

    if is_video:
        out_name = f"processed_{ts}_{uid}.mp4"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)

        out_path, graph_data_json, summary = pipeline.process_video(input_path, out_path, enabled=enabled)

        json_preview = json.dumps(summary, indent=2)

        # Save to history
        entry_id = str(uuid.uuid4())[:12]
        add_to_history({
            "id": entry_id,
            "original_filename": filename,
            "media_type": "video",
            "media_file": os.path.basename(out_path),
            "graph_data": graph_data_json,
            "json_preview": json_preview,
            "models_used": summary.get("models_used", []),
            "duration_sec": summary.get("duration_sec", 0),
            "total_objects": summary.get("total_objects_detected", 0),
            "frames": summary.get("frames", 0),
            "fps": summary.get("fps", 0),
            "resolution": summary.get("resolution", ""),
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        return render_template(
            "result.html",
            media_type="video",
            media_file=os.path.basename(out_path),
            graph_data=graph_data_json,
            json_preview=json_preview,
            fps=summary.get("fps", 25),
            total_frames=summary.get("frames", 0),
        )

    else:
        out_name = f"processed_{ts}_{uid}.png"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)

        out_path, summary = pipeline.process_image(input_path, out_path, enabled=enabled)

        json_preview = json.dumps(summary, indent=2)

        entry_id = str(uuid.uuid4())[:12]
        add_to_history({
            "id": entry_id,
            "original_filename": filename,
            "media_type": "image",
            "media_file": os.path.basename(out_path),
            "graph_data": None,
            "json_preview": json_preview,
            "models_used": summary.get("models_used", []),
            "duration_sec": 0,
            "total_objects": summary.get("objects", 0),
            "frames": 1,
            "fps": 0,
            "resolution": "",
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        return render_template(
            "result.html",
            media_type="image",
            media_file=os.path.basename(out_path),
            graph_data=None,
            json_preview=json_preview,
            fps=0,
            total_frames=0,
        )


# ===== History Routes =====

@app.route("/history")
def history():
    entries = load_history()
    return render_template("history.html", entries=entries)


@app.route("/history/<entry_id>")
def view_history(entry_id):
    entry = get_history_entry(entry_id)
    if not entry:
        return redirect(url_for('history'))

    return render_template(
        "result.html",
        media_type=entry.get('media_type'),
        media_file=entry.get('media_file'),
        graph_data=entry.get('graph_data'),
        json_preview=entry.get('json_preview'),
        fps=entry.get('fps', 25),
        total_frames=entry.get('frames', 0),
    )


@app.route("/history/<entry_id>/delete", methods=["POST"])
def delete_history(entry_id):
    hist = load_history()
    hist = [e for e in hist if e.get('id') != entry_id]
    save_history(hist)
    return redirect(url_for('history'))


@app.route("/history/clear", methods=["POST"])
def clear_history():
    save_history([])
    return redirect(url_for('history'))


@app.route("/settings")
def settings():
    return render_template("settings.html")


@app.route("/outputs/<filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    """Delete all files in uploads and outputs folders."""
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        for f in os.listdir(folder):
            fpath = os.path.join(folder, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
    return redirect(url_for('settings'))


if __name__ == "__main__":
    app.run(debug=True)
b
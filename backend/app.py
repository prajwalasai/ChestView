import os
import random

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # ...\ChestView\backend
PROJECT_DIR = os.path.dirname(BASE_DIR)                    # ...\ChestView
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "chestview_model.keras")

# ===== Settings =====
IMG_SIZE = (224, 224)   # must match the size used in training
MAX_UPLOAD_MB = 10

app = Flask(__name__, static_folder=None)
CORS(app)  # lets the page talk to the backend even if opened from another port
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

model = None


def load_model():
    """Loads the trained model if the file exists. Otherwise placeholder mode."""
    global model
    if os.path.exists(MODEL_PATH):
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded from", MODEL_PATH)
    else:
        print("No trained model found at", MODEL_PATH)
        print("Running in PLACEHOLDER mode (predictions are random).")


def prepare_image(file_stream):
    """Opens the uploaded image and turns it into the shape the model expects."""
    img = Image.open(file_stream)
    if img.mode.startswith("I"):                       # 16-bit or 32-bit grayscale PNG
        arr = np.array(img, dtype="float32")
        span = max(float(arr.max() - arr.min()), 1.0)
        arr = (arr - arr.min()) / span * 255.0
        img = Image.fromarray(arr.astype("uint8"))
    img = img.convert("RGB").resize(IMG_SIZE)          # X-rays can be grayscale
    arr = np.array(img, dtype="float32")               # pixel values 0 to 255
    return np.expand_dims(arr, axis=0)                 # shape (1, 224, 224, 3)


# ===== Website pages =====
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ===== API =====
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file was selected."}), 400

    try:
        batch = prepare_image(file.stream)
    except Exception:
        return jsonify({"error": "The file is not a valid image."}), 400

    # Placeholder mode (no trained model yet)
    if model is None:
        view = random.choice(["PA", "AP"])
        confidence = round(random.uniform(0.80, 0.99), 4)
        return jsonify({"view": view, "confidence": confidence, "demo": True})

    # Real model: output is the probability that the image is PA (0 to 1)
    prob_pa = float(model.predict(batch, verbose=0)[0][0])
    if prob_pa >= 0.5:
        view, confidence = "PA", prob_pa
    else:
        view, confidence = "AP", 1 - prob_pa

    return jsonify({"view": view, "confidence": round(confidence, 4), "demo": False})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Image is too large (max %d MB)." % MAX_UPLOAD_MB}), 413


if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
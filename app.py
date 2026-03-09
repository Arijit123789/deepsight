import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import base64
import io
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = "deepfake_mobilenetv3_attention.h5"
IMG_SIZE = 192
THRESHOLD = 0.65

print("Loading model...")

# ----------------------------
# MODEL
# ----------------------------

from tensorflow.keras.applications import MobileNetV3Small

input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

base_model = MobileNetV3Small(
    weights=None,
    include_top=False,
    input_tensor=input_tensor
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_tensor, outputs=output)

model.load_weights(MODEL_PATH)

print("Model loaded")

# ----------------------------
# FLASK
# ----------------------------

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return app.send_static_file("index.html")

# ----------------------------
# PREDICT
# ----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        img_b64 = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_b64)

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img = img.resize((IMG_SIZE, IMG_SIZE))

        arr = np.array(img).astype(np.float32)

        arr = preprocess_input(arr)

        arr = np.expand_dims(arr, axis=0)

        score = float(model.predict(arr)[0][0])

        is_fake = score > THRESHOLD

        confidence = score if not is_fake else (1 - score)

        return jsonify({

            "score": score,
            "label": "DEEPFAKE" if is_fake else "REAL",
            "is_fake": bool(is_fake),
            "confidence": round(confidence * 100, 2)

        })

    except Exception as e:

        return jsonify({"error": str(e)}), 500


# ----------------------------

if __name__ == "__main__":

    print("Server started")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

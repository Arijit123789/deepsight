import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import base64
import io
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from flask import Flask, request, jsonify
from flask_cors import CORS


# ─────────────────────────────────
# CONFIG
# ─────────────────────────────────

MODEL_PATH = "deepfake_mobilenetv3_attention.h5"
IMG_SIZE = 192
THRESHOLD = 0.65

# smoothing buffer
score_buffer = []

print("Building model...")

# ─────────────────────────────────
# CBAM ATTENTION BLOCK
# ─────────────────────────────────

def cbam_block(input_feature, ratio=8):

    channel = input_feature.shape[-1]

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Dense(channel // ratio, activation="relu")(avg_pool)
    avg_pool = Dense(channel, activation="sigmoid")(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Dense(channel // ratio, activation="relu")(max_pool)
    max_pool = Dense(channel, activation="sigmoid")(max_pool)

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Reshape((1,1,channel))(channel_attention)

    x = Multiply()([input_feature, channel_attention])

    avg_pool = Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
    max_pool = Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(x)

    concat = Concatenate(axis=-1)([avg_pool,max_pool])

    spatial_attention = Conv2D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid"
    )(concat)

    return Multiply()([x, spatial_attention])


# ─────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────

from tensorflow.keras.applications import MobileNetV3Small

input_tensor = Input(shape=(IMG_SIZE,IMG_SIZE,3))

base_model = MobileNetV3Small(
    weights=None,
    include_top=False,
    input_tensor=input_tensor
)

x = base_model.output
x = cbam_block(x)

x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(1, activation="sigmoid", dtype="float32")(x)

model = Model(inputs=input_tensor, outputs=output)

model.load_weights(MODEL_PATH)

print("Model loaded")

# warmup
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
model.predict(dummy)

# ─────────────────────────────────
# FACE DETECTOR
# ─────────────────────────────────

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ─────────────────────────────────
# GRADCAM HEATMAP
# ─────────────────────────────────

def gradcam(img_tensor):

    last_conv = None

    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            last_conv = layer.name
            break

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:,0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap,0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


# ─────────────────────────────────
# FLASK
# ─────────────────────────────────

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return app.send_static_file("index.html")


# ─────────────────────────────────
# PREDICT API
# ─────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():

    global score_buffer

    try:

        data = request.json

        img_b64 = data["image"].split(",")[1]

        img_bytes = base64.b64decode(img_b64)

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        frame = np.array(img)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:

            face = frame
            bbox = None

        else:

            x,y,w,h = faces[0]

            pad = int(0.2*w)

            x1 = max(0, x-pad)
            y1 = max(0, y-pad)
            x2 = min(frame.shape[1], x+w+pad)
            y2 = min(frame.shape[0], y+h+pad)

            face = frame[y1:y2, x1:x2]

            bbox = [int(x),int(y),int(w),int(h)]

        # lighting normalization
        face = cv2.cvtColor(face, cv2.COLOR_RGB2YCrCb)
        face[:,:,0] = cv2.equalizeHist(face[:,:,0])
        face = cv2.cvtColor(face, cv2.COLOR_YCrCb2RGB)

        face = cv2.resize(face,(IMG_SIZE,IMG_SIZE))

        arr = np.array(face, dtype=np.float32)

        arr_model = preprocess_input(arr.copy())
        arr_model = np.expand_dims(arr_model, axis=0)

        raw_score = float(model.predict(arr_model)[0][0])

        # temporal smoothing
        score_buffer.append(raw_score)

        if len(score_buffer) > 5:
            score_buffer.pop(0)

        smooth_score = float(np.mean(score_buffer))

        is_fake = smooth_score > THRESHOLD

        confidence = smooth_score if not is_fake else (1 - smooth_score)

        # GradCAM heatmap
        heatmap = gradcam(arr_model)

        heatmap = cv2.resize(heatmap,(IMG_SIZE,IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(face,0.6,heatmap,0.4,0)

        _,buffer = cv2.imencode(".jpg",overlay)

        heatmap_b64 = base64.b64encode(buffer).decode()

        return jsonify({

            "score": round(smooth_score,4),
            "label": "DEEPFAKE" if is_fake else "REAL",
            "is_fake": bool(is_fake),
            "confidence": round(confidence*100,1),
            "bbox": bbox,
            "attention": heatmap_b64

        })

    except Exception as e:

        return jsonify({"error":str(e)}),500


# ─────────────────────────────────
# IMAGE UPLOAD
# ─────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["image"]

    img = Image.open(file).convert("RGB")

    img = img.resize((IMG_SIZE,IMG_SIZE))

    arr = np.array(img)

    arr = preprocess_input(arr)

    arr = np.expand_dims(arr,0)

    score = float(model.predict(arr)[0][0])

    is_fake = score > THRESHOLD

    return jsonify({

        "score": score,
        "label": "DEEPFAKE" if is_fake else "REAL"

    })


# ─────────────────────────────────

if __name__ == "__main__":

    print("Server started")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=False
    )

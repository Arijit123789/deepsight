import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

import mediapipe as mp

MODEL_PATH = "deepfake_mobilenetv3_attention.h5"
IMG_SIZE = 192
THRESHOLD = 0.65

print("Loading model...")

# ------------------------
# MODEL
# ------------------------

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

# Warmup
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
model.predict(dummy)

# ------------------------
# MEDIAPIPE
# ------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

blink_frames = 0
blink_count = 0

# ------------------------
# EAR FUNCTION
# ------------------------

def eye_aspect_ratio(landmarks, eye_indices, w, h):

    pts=[]

    for i in eye_indices:
        x=int(landmarks[i].x*w)
        y=int(landmarks[i].y*h)
        pts.append((x,y))

    p2_p6=np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
    p3_p5=np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
    p1_p4=np.linalg.norm(np.array(pts[0])-np.array(pts[3]))

    ear=(p2_p6+p3_p5)/(2.0*p1_p4)

    return ear

# ------------------------
# FLASK
# ------------------------

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return app.send_static_file("index.html")

# ------------------------
# PREDICT
# ------------------------

@app.route("/predict", methods=["POST"])
def predict():

    global blink_frames
    global blink_count

    try:

        data = request.json

        img_b64 = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_b64)

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame = np.array(img)

        h,w,_=frame.shape

        # ------------------
        # BLINK DETECTION
        # ------------------

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results=face_mesh.process(rgb)

        blink_detected=False

        if results.multi_face_landmarks:

            landmarks=results.multi_face_landmarks[0].landmark

            leftEAR=eye_aspect_ratio(landmarks,LEFT_EYE,w,h)
            rightEAR=eye_aspect_ratio(landmarks,RIGHT_EYE,w,h)

            ear=(leftEAR+rightEAR)/2

            if ear<0.21:
                blink_frames+=1
            else:
                if blink_frames>2:
                    blink_count+=1
                    blink_detected=True
                blink_frames=0

        # ------------------
        # MODEL PREDICTION
        # ------------------

        face=cv2.resize(frame,(IMG_SIZE,IMG_SIZE))

        arr=preprocess_input(face.astype(np.float32))
        arr=np.expand_dims(arr,0)

        raw_score=float(model.predict(arr,verbose=0)[0][0])

        is_fake=raw_score>THRESHOLD

        confidence = raw_score if is_fake else (1-raw_score)

        # ------------------
        # BLINK ADJUSTMENT
        # ------------------

        if blink_count>=1:
            is_fake=False
            confidence=min(confidence+0.15,1.0)

        return jsonify({

            "score":round(raw_score,4),
            "label":"DEEPFAKE" if is_fake else "REAL",
            "is_fake":bool(is_fake),
            "confidence":round(confidence*100,2),
            "blinks":blink_count,
            "blink_detected":blink_detected

        })

    except Exception as e:
        return jsonify({"error":str(e)}),500

# ------------------------

if __name__=="__main__":

    print("Server started")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=False
    )

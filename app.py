import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

# ══════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════
MODEL_PATH = 'deepfake_mobilenetv3_attention.h5'
IMG_SIZE   = 192
THRESHOLD  = 0.5   # > 0.5 = FAKE

# ══════════════════════════════════════════
#  BUILD MODEL ARCHITECTURE + LOAD WEIGHTS
# ══════════════════════════════════════════
print("🔧 Building model architecture...")

def cbam_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Dense(channel // ratio, activation='relu')(avg_pool)
    avg_pool = Dense(channel, activation='sigmoid')(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Dense(channel // ratio, activation='relu')(max_pool)
    max_pool = Dense(channel, activation='sigmoid')(max_pool)

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Reshape((1, 1, channel))(channel_attention)
    x = Multiply()([input_feature, channel_attention])

    avg_pool = Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
    max_pool = Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(x)
    concat   = Concatenate(axis=-1)([avg_pool, max_pool])

    spatial_attention = Conv2D(
        filters=1, kernel_size=7,
        padding='same', activation='sigmoid'
    )(concat)

    x = Multiply()([x, spatial_attention])
    return x

from tensorflow.keras.applications import MobileNetV3Small
input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = MobileNetV3Small(weights=None, include_top=False, input_tensor=input_tensor)
base_model.trainable = True

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
print(f"✅ Model loaded from {MODEL_PATH}")

# Warmup
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
model.predict(dummy, verbose=0)
print("✅ Model warmed up — ready!")

# ══════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data  = request.json
        img_b64 = data['image'].split(',')[1]  # strip "data:image/jpeg;base64,"
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))

        arr = np.array(img, dtype=np.float32)

        # ── Screen artifact signals ──
        gray = np.mean(arr, axis=2)

        # Sharpness via Laplacian variance (blurry screen = low value)
        def lap_var(g):
            k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
            pad = np.pad(g, 1, mode='reflect')
            out = np.zeros_like(g)
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    out[i,j] = np.sum(pad[i:i+3, j:j+3] * k)
            return float(np.var(out))

        # Fast approximate sharpness using numpy gradients
        gy, gx = np.gradient(gray)
        sharpness = float(np.mean(gx**2 + gy**2))

        # Channel correlation (phone screen has different RGB balance)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        rg = float(np.corrcoef(r.flatten(), g.flatten())[0,1])
        rb = float(np.corrcoef(r.flatten(), b.flatten())[0,1])
        avg_corr = (rg + rb) / 2.0

        # ── Model prediction ──
        arr_model = preprocess_input(arr.copy())
        arr_model = np.expand_dims(arr_model, axis=0)
        raw_score = float(model.predict(arr_model, verbose=0)[0][0])

        print(f"Score: {raw_score:.4f} | Sharpness: {sharpness:.2f} | CorrRGB: {avg_corr:.3f}")

        # ── Decision logic based on observed data ──
        # Real face:    Sharpness 60–150,  CorrRGB 0.96–0.99, Score 0.95–1.00
        # Phone screen: Sharpness 370–560, CorrRGB 0.90–0.97, Score 0.55–0.99
        #
        # Best separator: Sharpness alone > 280 reliably = phone screen

        if sharpness > 280:
            # High sharpness = emitted light from screen = FAKE
            is_fake    = True
            confidence = min(0.99, 0.65 + (sharpness - 280) / 1000.0)
        elif sharpness < 200 and raw_score > 0.75:
            # Low sharpness + high model score = real face
            is_fake    = False
            confidence = raw_score
        else:
            # Ambiguous zone — trust model score with flipped label
            is_fake    = raw_score < THRESHOLD
            confidence = (1.0 - raw_score) if is_fake else raw_score

        return jsonify({
            'score':      round(raw_score, 4),
            'is_fake':    bool(is_fake),
            'label':      'DEEPFAKE' if is_fake else 'REAL',
            'confidence': round(float(confidence) * 100, 1),
            'sharpness':  round(sharpness, 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🌐 Open: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
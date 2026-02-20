import os
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import time
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow

# ── Load Model ─────────────────────────────────────────────────────────────────
model = load_model("model/emotion_model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model loaded!")

# ── Labels & Detector ─────────────────────────────────────────────────────────
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ── Smoothing ──────────────────────────────────────────────────────────────────
SMOOTH_FRAMES      = 8
prediction_history = []

def smooth_prediction(new_pred):
    prediction_history.append(new_pred)
    if len(prediction_history) > SMOOTH_FRAMES:
        prediction_history.pop(0)
    return np.mean(prediction_history, axis=0)

# ── Webcam Capture ─────────────────────────────────────────────────────────────
def capture_frame():
    js_code = """
        (async () => {
            const div = document.createElement('div');
            div.style.display = 'none';
            document.body.appendChild(div);
            const video = document.createElement('video');
            div.appendChild(video);
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            await video.play();
            await new Promise(r => setTimeout(r, 1500));
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks().forEach(t => t.stop());
            div.remove();
            return canvas.toDataURL('image/jpeg', 0.9);
        })()
    """
    data = eval_js(js_code)
    if not data or data == 'data:image/jpeg;base64,':
        return None
    img_bytes = base64.b64decode(data.split(',')[1])
    nparr     = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# ── Detection Loop ─────────────────────────────────────────────────────────────
NUM_FRAMES = 20
DELAY      = 0.3

print(f"Starting detection — {NUM_FRAMES} frames. Allow camera when prompted.\n")

for i in range(NUM_FRAMES):
    frame = capture_frame()
    if frame is None:
        print(f"Frame {i+1}: Could not capture.")
        continue

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
    )

    if len(faces) == 0:
        print(f"Frame {i+1}: No face detected.")
        cv2_imshow(frame)
        continue

    for (x, y, w, h) in faces:
        pad = int(0.15 * w)
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(frame.shape[1], x + w + pad)
        y2  = min(frame.shape[0], y + h + pad)

        face = frame[y1:y2, x1:x2]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))

        # ── Preprocessing matches training exactly ─────────────────────────────
        face = tf.keras.applications.mobilenet_v2.preprocess_input(
            face.astype(np.float32)
        )
        face = np.expand_dims(face, axis=0)

        raw_pred   = model.predict(face, verbose=0)[0]
        prediction = smooth_prediction(raw_pred)
        max_idx    = np.argmax(prediction)
        confidence = prediction[max_idx]

        # Uncertainty threshold
        if confidence < 0.35:
            emotion = "Uncertain"
            color   = (128, 128, 128)
        else:
            emotion = emotion_labels[max_idx]
            color   = (0, 220, 0) if confidence > 0.60 else \
                      (0, 165, 255) if confidence > 0.40 else (0, 0, 220)

        # Draw box + label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{emotion}: {confidence*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)
        cv2.putText(frame, label, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Confidence bars
        bar_x, bar_y = x, y + h + 5
        bar_w = max(w, 180)
        bar_h = 13
        for j, (lbl, prob) in enumerate(zip(emotion_labels, prediction)):
            filled = int(prob * bar_w)
            row_y  = bar_y + j * (bar_h + 3)
            cv2.rectangle(frame, (bar_x, row_y), (bar_x + bar_w, row_y + bar_h), (40, 40, 40), -1)
            bc = color if j == max_idx else (110, 110, 110)
            cv2.rectangle(frame, (bar_x, row_y), (bar_x + filled, row_y + bar_h), bc, -1)
            cv2.putText(frame, f"{lbl[:3]} {prob*100:.0f}%",
                        (bar_x + 2, row_y + bar_h - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)

        print(f"Frame {i+1}: {emotion} ({confidence*100:.1f}%)")

    cv2_imshow(frame)
    time.sleep(DELAY)

print("\nDetection complete!")

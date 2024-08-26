import os

from PIL import Image

import numpy as np
import torch
from common import utils
from ultralytics import YOLO
from transformers import AutoImageProcessor

from flask import Flask, request, jsonify, render_template
from transformers import AutoImageProcessor
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)

# Useful constants
CURRENT_DIR = os.getcwd()
IMAGES_DIR = os.path.join(CURRENT_DIR, "images")
VIDEOS_DIR = os.path.join(CURRENT_DIR, "videos")
CHORD_CLASSIFIER_MODEL_DIR = os.path.join(CURRENT_DIR, "chord-classifier-model")
FRETBOARD_RECOGNIZER_MODEL_DIR = os.path.join(CURRENT_DIR, "fretboard-recognizer-model")

chord_clf_model_path = utils.find_files(CHORD_CLASSIFIER_MODEL_DIR, [".safetensors", ".pt"])
chord_clf_config_path = utils.find_files(CHORD_CLASSIFIER_MODEL_DIR, [".json"])
fretboard_rec_model_path = utils.find_files(FRETBOARD_RECOGNIZER_MODEL_DIR, [".safetensors", ".pt"])
fretboard_rec_config_path = utils.find_files(FRETBOARD_RECOGNIZER_MODEL_DIR, [".json"])

utils.ensure_files_exist(
    chord_clf_model_path,
    fretboard_rec_model_path,
    chord_clf_config_path,
    fretboard_rec_config_path,
    names=[
        "Chord Classifier model",
        "Fretboard Recognizer model",
        "Chord Classifier config",
        "Fretboard Recognizer config",
    ],
)

# Load Chord Classifier model
chord_clf_model = utils.load_model(chord_clf_model_path, config_path=chord_clf_config_path)
chord_clf_model.eval()

# Load Fretboard Recognizer model
fretboard_rec_model = utils.load_model(fretboard_rec_model_path, config_path=fretboard_rec_config_path, custom_class=YOLO)

feature_extractor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

print("Models loaded successfully.")

fps = 30

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_video_frame():
    # Get the base64 encoded image from the request
    data = request.json
    image_data = data['image']
    
    # Decode the base64 image
    image_bytes = base64.b64decode(image_data)
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Perform inference on the fretboard recognizer model
    results = fretboard_rec_model.predict(pil_image)[0].boxes
    indices = (results.cls == 80).nonzero(as_tuple=True)[0]

    if len(indices) > 0:
        # Get the bounding box with the highest confidence
        max_conf_index = results.conf[indices].argmax()
        result = results.data[indices[max_conf_index]]

        # Increase bounding box size by 15%
        x1, y1, x2, y2 = result[:4]
        width = x2 - x1
        height = y2 - y1
        increase_x = width * 0.90 / 2
        increase_y = height * 0.90 / 2

        new_x1 = max(0, x1 - increase_x)
        new_y1 = max(0, y1 - increase_y)
        new_x2 = min(pil_image.width, x2 + increase_x)
        new_y2 = min(pil_image.height, y2 + increase_y)

        # Crop the fretboard with increased bounding box
        pil_image = pil_image.crop(np.array([new_x1, new_y1, new_x2, new_y2]))

    # Preprocess the image
    inputs = feature_extractor(images=pil_image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = chord_clf_model(**inputs)

    # Get the predicted class
    predicted_class_idx = outputs.logits.argmax(-1).item()
    predicted_class = chord_clf_model.config.id2label[predicted_class_idx]
    
    # Return the result
    return jsonify({"class": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Path expected: models/tomato_custom_cnn/model.h5
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "tomato_custom_cnn", "model.h5"
)

IMG_SIZE = (160, 160)

# Urutan kelas HARUS sama persis dengan urutan saat training
# (hasil dari sorted(os.listdir(train_dir)) di notebook)
CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


@st.cache_resource
def load_tomato_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32) / 255.0  # sama seperti rescale=1./255 saat training
    return np.expand_dims(array, axis=0)


def predict_tomato(model, uploaded_image):
    batch = preprocess_image(uploaded_image)
    probs = model.predict(batch, verbose=0)[0]

    predicted_idx = int(np.argmax(probs))
    label = CLASSES[predicted_idx]
    confidence = float(probs[predicted_idx])
    all_probs = {cls: float(p) for cls, p in zip(CLASSES, probs)}

    return label, confidence, all_probs

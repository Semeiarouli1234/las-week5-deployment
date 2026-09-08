import os
import re

import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Path expected: models/indobert_sentiment/ (folder hasil trainer.save_model()
# + tokenizer.save_pretrained(), berisi config.json, model.safetensors, dst.)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "indobert_sentiment")

MAX_LEN = 128

# Urutan label HARUS sama persis dengan label_map saat training
ID2LABEL = {0: "Positive", 1: "Neutral", 2: "Negative"}

SLANG_DICT = {
    "yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "tdk": "tidak", "bgt": "banget", "dr": "dari", "dgn": "dengan", "utk": "untuk",
    "sdh": "sudah", "udh": "sudah", "blm": "belum", "jd": "jadi", "krn": "karena",
    "karna": "karena", "spy": "supaya", "kalo": "kalau", "klo": "kalau",
    "gmn": "bagaimana", "gimana": "bagaimana", "emg": "memang", "emang": "memang",
    "trs": "terus", "tp": "tapi", "org": "orang", "orng": "orang", "sm": "sama",
    "jgn": "jangan", "gini": "begini", "gitu": "begitu", "aja": "saja", "doang": "saja",
    "bs": "bisa", "skrg": "sekarang", "sekrg": "sekarang", "moga": "semoga", "smg": "semoga",
    "dpt": "dapat", "dapet": "dapat", "bnyk": "banyak", "byk": "banyak",
    "knp": "kenapa", "gapapa": "tidak apa apa", "gpp": "tidak apa apa", "mksh": "terima kasih",
    "makasih": "terima kasih", "pemrintah": "pemerintah", "gzi": "gizi", "mkn": "makan",
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [SLANG_DICT.get(w, w) for w in words]
    text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_sentiment_model():
    if not os.path.isdir(MODEL_DIR):
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


def predict_sentiment(model, tokenizer, text):
    cleaned = clean_text(text)
    encoding = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    predicted_idx = int(np.argmax(probs))
    label = ID2LABEL[predicted_idx]
    confidence = float(probs[predicted_idx])
    all_probs = {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}

    return label, confidence, all_probs

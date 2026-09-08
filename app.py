import streamlit as st

from utils.tomato_model import load_tomato_model, predict_tomato
from utils.sentiment_model import load_sentiment_model, predict_sentiment

st.set_page_config(page_title="LAS Big Data - Model Deployment", layout="centered")

st.title("Deployment Model LAS Big Data")
st.caption("Custom CNN (Tomato Leaf Disease) + IndoBERT (Sentimen MBG)")

tab_tomato, tab_sentiment, tab_versioning = st.tabs(
    ["Klasifikasi Daun Tomat", "Analisis Sentimen MBG", "Versioning"]
)

with tab_tomato:
    st.subheader("Klasifikasi Penyakit Daun Tomat")
    st.write(
        "Upload foto daun tomat, model akan memprediksi jenis penyakit "
        "(atau kondisi sehat) dari 10 kelas yang tersedia."
    )

    tomato_model = load_tomato_model()

    uploaded_image = st.file_uploader(
        "Upload gambar daun tomat", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Gambar yang diupload", width=300)

        if st.button("Prediksi", key="predict_tomato"):
            if tomato_model is None:
                st.error(
                    "Model belum tersedia. Taruh file model di "
                    "`models/tomato_mobilenetv2_ft/` lalu jalankan ulang."
                )
            else:
                with st.spinner("Memproses..."):
                    label, confidence, all_probs = predict_tomato(
                        tomato_model, uploaded_image
                    )
                st.success(f"Prediksi: **{label}** ({confidence:.1%})")
                st.bar_chart(all_probs)

with tab_sentiment:
    st.subheader("Analisis Sentimen Komentar MBG")
    st.write(
        "Masukkan komentar terkait program Makan Bergizi Gratis (MBG), "
        "model akan memprediksi sentimennya: Positive, Neutral, atau Negative."
    )

    sentiment_model, sentiment_tokenizer = load_sentiment_model()

    text_input = st.text_area("Komentar", placeholder="Tulis komentar di sini...")

    if st.button("Prediksi", key="predict_sentiment"):
        if not text_input.strip():
            st.warning("Komentar tidak boleh kosong.")
        elif sentiment_model is None:
            st.error(
                "Model belum tersedia. Taruh file model di "
                "`models/indobert_sentiment/` lalu jalankan ulang."
            )
        else:
            with st.spinner("Memproses..."):
                label, confidence, all_probs = predict_sentiment(
                    sentiment_model, sentiment_tokenizer, text_input
                )
            st.success(f"Prediksi: **{label}** ({confidence:.1%})")
            st.bar_chart(all_probs)

with tab_versioning:
    st.subheader("Riwayat Versi Model")

    st.markdown("**Model 1: Klasifikasi Daun Tomat**")
    st.table(
        {
            "Versi": ["v1", "v2", "v-final"],
            "Model": [
                "MobileNetV2 (Frozen)",
                "MobileNetV2 (Fine-Tuned)",
                "Custom CNN",
            ],
            "Accuracy": ["86.8%", "93.8%", "94.8%"],
            "F1 (macro)": [0.868, 0.938, 0.948],
            "Catatan": [
                "Transfer learning, base frozen",
                "Fine-tuning 50 layer teratas, naik dari v1",
                "CNN custom 4 blok konvolusi, hasil terbaik",
            ],
        }
    )

    st.markdown("**Model 2: Analisis Sentimen MBG**")
    st.table(
        {
            "Versi": ["v1", "v2", "v-final"],
            "Model": [
                "TF-IDF + Logistic Regression",
                "TF-IDF + Linear SVM",
                "IndoBERT Fine-tuned",
            ],
            "F1 (macro)": [0.740, 0.787, 0.895],
            "Catatan": [
                "Baseline linear model dengan class_weight balanced",
                "Peningkatan dari kernel linear SVM",
                "Fine-tuning indobenchmark/indobert-base-p1, hasil terbaik",
            ],
        }
    )

    st.info(
        "Screenshot/dokumentasi tiap versi sebelumnya bisa ditambahkan di sini "
        "sebagai gambar (st.image) sesuai kriteria penilaian tugas."
    )

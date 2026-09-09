import streamlit as st

from utils.tomato_model import load_tomato_model, predict_tomato
from utils.sentiment_model import load_sentiment_model, predict_sentiment

st.set_page_config(page_title="LAS Big Data - Model Deployment", layout="centered")


def prettify_tomato_label(label: str) -> str:
    """Ubah 'Tomato___Late_blight' jadi 'Late Blight' agar enak dibaca."""
    name = label.replace("Tomato___", "").replace("_", " ").strip()
    return name.title()


st.title("Deployment Model LAS Big Data")
st.caption("Custom CNN (Tomato Leaf Disease) + IndoBERT (Sentimen MBG)")

tab_tomato, tab_sentiment, tab_versioning = st.tabs(
    ["Klasifikasi Daun Tomat", "Analisis Sentimen MBG", "Versioning"]
)

with tab_tomato:
    st.subheader("🍅 Klasifikasi Penyakit Daun Tomat")
    st.write(
        "Upload foto daun tomat, model akan memprediksi jenis penyakit "
        "(atau kondisi sehat) dari 10 kelas yang tersedia."
    )

    tomato_model = load_tomato_model()

    col_upload, col_result = st.columns([1, 1.2], gap="large")

    with col_upload:
        uploaded_image = st.file_uploader(
            "Upload gambar daun tomat", type=["jpg", "jpeg", "png"]
        )
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Gambar yang diupload", use_container_width=True)
            predict_clicked = st.button(
                "🔍 Prediksi", key="predict_tomato", use_container_width=True
            )
        else:
            st.info("Upload gambar dulu untuk mulai prediksi.")
            predict_clicked = False

    with col_result:
        if uploaded_image is not None and predict_clicked:
            if tomato_model is None:
                st.error(
                    "Model belum tersedia. Taruh file model di "
                    "`models/tomato_custom_cnn/` lalu jalankan ulang."
                )
            else:
                with st.spinner("Menganalisis gambar..."):
                    label, confidence, all_probs = predict_tomato(
                        tomato_model, uploaded_image
                    )

                display_label = prettify_tomato_label(label)
                is_healthy = "healthy" in label.lower()

                if is_healthy:
                    st.success(f"### ✅ {display_label}")
                else:
                    st.warning(f"### 🩺 {display_label}")
                st.caption(f"Tingkat keyakinan model: **{confidence:.1%}**")

                st.markdown("**Top 3 kemungkinan:**")
                top3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                for cls, prob in top3:
                    st.write(prettify_tomato_label(cls))
                    st.progress(prob, text=f"{prob:.1%}")
        elif uploaded_image is None:
            st.empty()

with tab_sentiment:
    st.subheader("💬 Analisis Sentimen Komentar MBG")
    st.write(
        "Masukkan komentar terkait program Makan Bergizi Gratis (MBG), "
        "model akan memprediksi sentimennya: Positive, Neutral, atau Negative."
    )

    sentiment_model, sentiment_tokenizer = load_sentiment_model()

    with st.expander("💡 Coba contoh komentar"):
        examples = [
            "Program ini bagus banget buat anak sekolah, gizinya jadi terjamin",
            "Menunya kadang gak sesuai jadwal, kurang konsisten",
            "Biasa aja sih, gak terlalu berpengaruh ke anak saya",
        ]
        for ex in examples:
            if st.button(ex, key=f"example_{ex[:15]}"):
                st.session_state["komentar_input"] = ex

    text_input = st.text_area(
        "Komentar",
        placeholder="Tulis komentar di sini...",
        key="komentar_input",
    )

    predict_sentiment_clicked = st.button("🔍 Prediksi", key="predict_sentiment")

    if predict_sentiment_clicked:
        if not text_input.strip():
            st.warning("Komentar tidak boleh kosong.")
        elif sentiment_model is None:
            st.error(
                "Model belum tersedia. Taruh file model di "
                "`models/indobert_sentiment/` lalu jalankan ulang."
            )
        else:
            with st.spinner("Menganalisis komentar..."):
                label, confidence, all_probs = predict_sentiment(
                    sentiment_model, sentiment_tokenizer, text_input
                )

            icon_map = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
            color_map = {"Positive": "success", "Neutral": "info", "Negative": "error"}
            icon = icon_map.get(label, "")
            getattr(st, color_map.get(label, "info"))(
                f"### {icon} {label}  \nTingkat keyakinan: **{confidence:.1%}**"
            )

            st.markdown("**Distribusi probabilitas tiap kelas:**")
            for cls in ["Positive", "Neutral", "Negative"]:
                prob = all_probs.get(cls, 0.0)
                st.write(f"{icon_map.get(cls, '')} {cls}")
                st.progress(prob, text=f"{prob:.1%}")

with tab_versioning:
    st.subheader("Riwayat Versi Model")
    st.caption(
        "Pencatatan progres tiap model dari versi awal sampai versi final yang di-deploy."
    )

    # ---------- Model 1: Tomato Leaf ----------
    st.markdown("### 🍅 Klasifikasi Daun Tomat")

    col1, col2, col3 = st.columns(3)
    col1.metric("v1 — MobileNetV2 (Frozen)", "86.8%", help="F1 macro: 0.868")
    col2.metric(
        "v2 — MobileNetV2 (Fine-Tuned)", "93.8%", "+7.0%", help="F1 macro: 0.938"
    )
    col3.metric(
        "v-final — Custom CNN", "94.8%", "+1.0%", help="F1 macro: 0.948 (terbaik)"
    )

    with st.expander("Detail riwayat versi Tomato Leaf"):
        st.markdown(
            """
- **v1 — MobileNetV2 (Frozen):** transfer learning dengan base model dibekukan,
  jadi fitur bawaan ImageNet dipakai langsung tanpa disesuaikan ke gambar daun tomat.
- **v2 — MobileNetV2 (Fine-Tuned):** 50 layer teratas base model dibuka dan
  dilatih ulang, akurasi naik cukup signifikan dari v1.
- **v-final — Custom CNN:** arsitektur CNN dirancang sendiri (4 blok konvolusi),
  hasilnya melampaui MobileNetV2 dan dipilih sebagai model yang di-deploy.
            """
        )

    img1, img2 = st.columns(2)
    with img1:
        st.image(
            "assets/tomato_comparison_table.png",
            caption="Tabel perbandingan ketiga versi",
        )
    with img2:
        st.image(
            "assets/tomato_comparison_chart.png",
            caption="Grafik accuracy & F1 score",
        )

    st.divider()

    # ---------- Model 2: Sentimen MBG ----------
    st.markdown("### 💬 Analisis Sentimen MBG")

    col4, col5, col6 = st.columns(3)
    col4.metric("v1 — TF-IDF + LogReg", "74.0%", help="F1 macro: 0.740")
    col5.metric("v2 — TF-IDF + SVM", "78.7%", "+4.7%", help="F1 macro: 0.787")
    col6.metric(
        "v-final — IndoBERT", "89.5%", "+10.8%", help="F1 macro: 0.895 (terbaik)"
    )

    with st.expander("Detail riwayat versi Sentimen MBG"):
        st.markdown(
            """
- **v1 — TF-IDF + Logistic Regression:** model linear baseline dengan
  `class_weight="balanced"` untuk menangani ketidakseimbangan kelas.
- **v2 — TF-IDF + Linear SVM:** ganti algoritma ke SVM kernel linear,
  performa naik dari baseline.
- **v-final — IndoBERT Fine-tuned:** fine-tuning model bahasa
  `indobenchmark/indobert-base-p1`, hasil jauh melampaui model linear
  dan dipilih sebagai model yang di-deploy.
            """
        )

    st.image(
        "assets/sentiment_comparison_table.png",
        caption="Tabel perbandingan ketiga versi model sentimen",
        width=500,
    )
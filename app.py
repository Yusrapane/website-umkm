from io import BytesIO
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Prediksi UMKM dengan LightGBM",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "model_lightgbm_umkm.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("📊 Sistem Prediksi Kategori UMKM")
st.write(
    "Unggah data UMKM dalam format Excel untuk memprediksi "
    "kategori usaha menggunakan model LightGBM."
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

# Ambil nama fitur yang digunakan saat model dilatih
if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    feature_cols = [
        "Monthly_Revenue",
        "Net_Profit_Margin (%)",
        "Burn_Rate_Ratio",
        "Transaction_Count",
        "Avg_Historical_Rating",
        "Review_Volatility",
        "Business_Tenure_Months",
        "Repeat_Order_Rate (%)",
        "Digital_Adoption_Score",
        "Peak_Hour_Latency",
        "Location_Competitiveness",
        "Sentiment_Score",
    ]

st.subheader("1. Unggah dataset UMKM")
uploaded_file = st.file_uploader(
    "Pilih file Excel (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        st.write("Ukuran data:", df.shape)
        st.write("Pratinjau data:")
        st.dataframe(df.head(10), use_container_width=True)

        missing_cols = [
            col for col in feature_cols if col not in df.columns
        ]

        if missing_cols:
            st.error(
                "Kolom berikut belum tersedia: "
                + ", ".join(missing_cols)
            )
            st.info(
                "Pastikan nama kolom sama dengan dataset training. "
                "Kolom target Class tidak perlu disertakan."
            )
        elif st.button("Prediksi Kategori UMKM", type="primary"):
            # Ambil fitur yang dibutuhkan, dengan urutan yang benar
            X_new = df[feature_cols].copy()

            with st.spinner("Model sedang melakukan prediksi..."):
                predictions = model.predict(X_new)

            hasil = df.copy()
            hasil["Prediksi_Class"] = predictions

            # Tampilkan probabilitas jika didukung model
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_new)
                classes = model.classes_

                hasil["Probabilitas_Tertinggi (%)"] = (
                    probabilities.max(axis=1) * 100
                ).round(2)

                hasil["Prediksi_Class"] = predictions

            st.success("Prediksi selesai!")
            st.subheader("2. Hasil prediksi")
            st.dataframe(hasil, use_container_width=True)

            st.subheader("3. Ringkasan kategori")
            st.bar_chart(hasil["Prediksi_Class"].value_counts())

            # Ekspor hasil ke Excel

            buffer = BytesIO()

            hasil.to_excel(
                buffer,
                index=False,
                engine="openpyxl"
            )

            st.download_button(
                label="Unduh hasil prediksi Excel",
                data=buffer.getvalue(),
                file_name="hasil_prediksi_umkm.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )


    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")

st.caption(
    "Hasil prediksi merupakan estimasi model, bukan jaminan "
    "keberhasilan atau kegagalan usaha."
)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ----------------------------
# 1. LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("spmb2023.csv")

df = load_data()

# ----------------------------
# 2. SIDEBAR
# ----------------------------
st.sidebar.title("Dashboard SPMB 2023")
menu = st.sidebar.radio("Pilih Halaman:", ["EDA", "Model", "Prediksi"])

# ----------------------------
# 3. HALAMAN 1: EDA
# ----------------------------
if menu == "EDA":
    st.title("📊 Eksplorasi Data (EDA)")

    st.write("### Cuplikan Data")
    st.dataframe(df.head())

    st.write("### Statistik Deskriptif")
    st.write(df.describe())

    st.write("### Korelasi antar Variabel")
    fig, ax = plt.subplots()
    sns.heatmap(df.drop(columns=["lokasi.formasi"]).corr(), annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

# ----------------------------
# 4. HALAMAN 2: TRAIN MODEL
# ----------------------------
elif menu == "Model":
    st.title("🤖 Pelatihan Model Regresi")

    X = df[["formasi.d3st", "formasi.d4st", "formasi.d4ks"]]
    y = df["pendaftar.d4ks"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("### Evaluasi Model")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Simpan model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/regresi_pendaftar_d4ks.pkl")
    st.success("✅ Model disimpan sebagai `regresi_pendaftar_d4ks.pkl`")

# ----------------------------
# 5. HALAMAN 3: PREDIKSI
# ----------------------------
elif menu == "Prediksi":
    st.title("📥 Prediksi Jumlah Pendaftar D4-KS")

    # Load model
    model_path = "model/regresi_pendaftar_d4ks.pkl"
    if not os.path.exists(model_path):
        st.warning("⚠️ Model belum dilatih. Silakan buka halaman 'Model' terlebih dahulu.")
    else:
        model = joblib.load(model_path)

        st.write("### Masukkan Jumlah Formasi")
        d3st = st.number_input("Formasi D3 Statistika", min_value=0, value=2)
        d4st = st.number_input("Formasi D4 Statistika", min_value=0, value=6)
        d4ks = st.number_input("Formasi D4 Komputasi Statistik", min_value=0, value=3)

        if st.button("Prediksi"):
            input_data = pd.DataFrame([[d3st, d4st, d4ks]], columns=["formasi.d3st", "formasi.d4st", "formasi.d4ks"])
            hasil = model.predict(input_data)
            st.success(f"🎯 Estimasi Jumlah Pendaftar D4-KS: **{hasil[0]:.0f} orang**")

# Input interaktif
name = st.text_input ("Siapa nama Anda?")
if name:
    st.succes(f"Halo, {name}!")

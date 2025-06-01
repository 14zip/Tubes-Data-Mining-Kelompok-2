import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('linear_regression_model.pkl')
        scaler = joblib.load('minmax_scaler.pkl')
        X_train_cols = joblib.load('X_train_cols.pkl')
        return model, scaler, X_train_cols
    except FileNotFoundError:
        st.error("File model, scaler, atau kolom tidak ditemukan. Pastikan Anda telah mengunggah file-file ini.")
        return None, None, None

model, scaler, X_train_cols = load_artifacts()

if model and scaler and X_train_cols:
    st.title("Dashboard Prediksi Tingkat Depresi Mahasiswa")
    st.sidebar.header("Input Data Mahasiswa")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 10, 100, 20)
    academic_pressure = st.sidebar.slider("Academic Pressure", 0, 100, 50)
    work_pressure = st.sidebar.slider("Work Pressure", 0, 100, 50)
    cgpa = st.sidebar.slider("CGPA", 0.0, 4.0, 3.0)
    financial_stress = st.sidebar.slider("Financial Stress", 0, 100, 50)
    suicidal_thoughts = st.sidebar.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
    family_history = st.sidebar.selectbox("Family History of Mental Illness", ["Yes", "No"])
    city = st.sidebar.selectbox("City", ["City A", "City B", "City C"])
    profession = st.sidebar.selectbox("Profession", ["Student", "Employed", "Other"])
    degree = st.sidebar.selectbox("Degree", ["Undergraduate", "Graduate", "PhD"])
    sleep_duration = st.sidebar.selectbox("Sleep Duration", ["Short", "Normal", "Long"])
    dietary_habits = st.sidebar.selectbox("Dietary Habits", ["Healthy", "Unhealthy"])

    input_data = {
        'Gender': gender,
        'Age': age,
        'Academic Pressure': academic_pressure,
        'Work Pressure': work_pressure,
        'CGPA': cgpa,
        'Financial Stress': financial_stress,
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Family History of Mental Illness': family_history,
        'City': city,
        'Profession': profession,
        'Degree': degree,
        'Sleep Duration': sleep_duration,
        'Dietary Habits': dietary_habits
    }

    input_df = pd.DataFrame([input_data])
    input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1})
    input_df['Have you ever had suicidal thoughts ?'] = input_df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
    input_df['Family History of Mental Illness'] = input_df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})

    input_df = pd.get_dummies(input_df, columns=['City', 'Profession', 'Degree', 'Sleep Duration', 'Dietary Habits'])

    for col in X_train_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X_train_cols]

    if st.sidebar.button("Prediksi"):
        prediction = model.predict(input_df)
        prediction = np.clip(prediction, 1, 10)  # Batasi hasil antara 1–10

        st.subheader("Hasil Prediksi Tingkat Depresi:")
        st.write(f"Tingkat Depresi yang Diprediksi: {prediction[0]:.2f}")
        st.markdown("---")
        st.write("Dashboard ini memberikan prediksi tingkat depresi berdasarkan data input.")
else:
    st.warning("Tidak dapat memuat file model, scaler, atau kolom. Pastikan file-file tersebut ada.")

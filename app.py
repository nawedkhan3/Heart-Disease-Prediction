import streamlit as st
import pandas as pd
import joblib

# Load artifacts created by Heart.ipynb
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# These must match the notebook's preprocessing exactly.
numeric_columns = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

st.title("Heart Disease Prediction")
st.caption(
    "Educational demo only. This model cannot diagnose heart disease "
    "or replace advice from a qualified healthcare professional."
)

st.markdown("Enter the details below:")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 250, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 700, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 50, 250, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 10.0, 1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    # Start with the same raw columns used in the notebook BEFORE get_dummies.
    raw_input = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }])

    # Match notebook: pd.get_dummies(df, drop_first=True)
    input_df = pd.get_dummies(raw_input, drop_first=True)
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Notebook fitted the scaler ONLY on these five numeric columns.
    # Do not pass the entire one-hot-encoded frame to scaler.transform().
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
         st.error("⚠️ High Risk of Heart Disease")
    else:
       st.success("✅ Low Risk of Heart Disease")

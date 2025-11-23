import streamlit as st
import joblib
import pandas as pd

# Charger le modèle
model = joblib.load("Model.pkl")

st.title("Prédiction du risque de CHD")

st.write("Veuillez saisir les informations du patient :")

sbp = st.number_input("Tension artérielle systolique (sbp)", 80.0, 250.0, 130.0)
ldl = st.number_input("LDL (ldl)", 0.0, 1000.0, 130.0)
adiposity = st.number_input("Adiposity", 0.0, 100.0, 25.0)
obesity = st.number_input("Obesity", 10.0, 60.0, 30.0)
age = st.number_input("Âge", 20.0, 100.0, 50.0)

famhist = st.selectbox(
    "Antécédents familiaux de maladie cardiaque (famhist)",
    ["Absent", "Present"]
)

data_input = pd.DataFrame({
    "sbp": [sbp],
    "ldl": [ldl],
    "adiposity": [adiposity],
    "famhist": [famhist],
    "obesity": [obesity],
    "age": [age]
})

if st.button("Prédire le risque"):
    pred = model.predict(data_input)[0]
    proba = model.predict_proba(data_input)[0][1]

    if pred == 1:
        st.error(f"Risque élevé de CHD\nProbabilité : {proba:.2f}")
    else:
        st.success(f"Risque faible de CHD\nProbabilité : {proba:.2f}")

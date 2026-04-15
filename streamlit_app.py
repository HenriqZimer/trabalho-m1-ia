from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "melhor_modelo.keras"
MODEL_FALLBACK_PATH = ARTIFACTS_DIR / "mlp_drybean.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler_final.pkl"
ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
ETAPA1_PATH = ARTIFACTS_DIR / "etapa1_artifacts.json"


@st.cache_resource
def load_objects():
    model_path = MODEL_PATH if MODEL_PATH.exists() else MODEL_FALLBACK_PATH
    model = keras.models.load_model(model_path)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    if ETAPA1_PATH.exists():
        with open(ETAPA1_PATH, "r", encoding="utf-8") as f:
            etapa1 = json.load(f)
        selected_features = etapa1.get("selected_features", [])
        log_candidates = etapa1.get("log_candidates", [])
    else:
        selected_features = [
            "Area",
            "MinorAxisLength",
            "Eccentricity",
            "Extent",
            "Solidity",
            "Roundness",
            "ShapeFactor1",
            "ShapeFactor3",
        ]
        log_candidates = []

    return model, scaler, encoder, selected_features, log_candidates


st.set_page_config(page_title="Dry Bean Classifier", layout="wide")
st.title("Dry Bean - Predicao de Classe")
st.write("Etapa 6: deploy simples em Streamlit usando modelo, scaler e encoder salvos.")

missing = []
if not (MODEL_PATH.exists() or MODEL_FALLBACK_PATH.exists()):
    missing.append(f"{MODEL_PATH} (ou {MODEL_FALLBACK_PATH})")

for p in [SCALER_PATH, ENCODER_PATH]:
    if not p.exists():
        missing.append(str(p))

if missing:
    st.error("Arquivos obrigatorios nao encontrados em artifacts/:\n- " + "\n- ".join(missing))
    st.stop()

model, scaler, encoder, selected_features, log_candidates = load_objects()

st.subheader("Entrada das features")
st.caption("Use valores brutos do dataset. O app aplica log1p quando necessario e depois normaliza com o scaler salvo.")
st.sidebar.header("Entradas")
st.sidebar.caption("Preencha os valores das features")

feature_values = {}
for feature in selected_features:
    feature_values[feature] = st.sidebar.number_input(
        label=feature,
        min_value=0.0,
        value=1.0,
        step=0.1,
        format="%.6f",
    )

if st.button("Prever classe", type="primary"):
    X_input = pd.DataFrame([feature_values], columns=selected_features)

    for col in selected_features:
        if col in log_candidates:
            X_input[col] = np.log1p(X_input[col])

    X_scaled = scaler.transform(X_input)
    probas = model.predict(X_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(probas))
    pred_label = encoder.inverse_transform([pred_idx])[0]

    st.success(f"Classe prevista: {pred_label}")

    prob_df = pd.DataFrame(
        {
            "Classe": encoder.classes_,
            "Probabilidade": probas,
        }
    ).sort_values("Probabilidade", ascending=False)

    st.subheader("Probabilidades por classe")
    st.bar_chart(prob_df.set_index("Classe"))
    st.dataframe(prob_df, use_container_width=True)

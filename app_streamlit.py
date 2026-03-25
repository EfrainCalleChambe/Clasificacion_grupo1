import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# ─────────────────────────────────────────────
# 01. Cargar el modelo
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return load('modelo_rf_enaho_2024_binario.joblib')

clf = load_model()

# ─────────────────────────────────────────────
# 02. Opciones de los selectbox
# ─────────────────────────────────────────────
dominio_options = [
    'costa norte', 'costa centro', 'costa sur',
    'sierra norte', 'sierra centro', 'sierra sur',
    'selva', 'lima metropolitana'
]

# El modelo fue entrenado con estrato como texto (ver notebook, celda 26)
estrato_options = {
    '1 – Menos de 401 viviendas':             'menos de 401 viviendas',
    '2 – De 401 a 1 200 viviendas':           'de 401 a 1200 viviendas',
    '3 – De 1 201 a 2 500 viviendas':         'de 1201 a 2500 viviendas',
    '4 – De 2 501 a 5 000 viviendas':         'de 2501 a 5000 viviendas',
    '5 – De 5 001 a 10 000 viviendas':        'de 5001 a 10000 viviendas',
    '6 – De 10 001 a 20 000 viviendas':       'de 10001 a 20000 viviendas',
    '7 – De 20 001 a 49 999 habitantes':      'de 20 000 a 49 999 habitantes',
    '8 – 50 000 habitantes o más':            '50000 o mas habitantes',
}

# ─────────────────────────────────────────────
# 03. Interfaz
# ─────────────────────────────────────────────
st.title("Modelo Predictivo de Pobreza – ENAHO 2024")
st.markdown(
    "Este modelo predice si un hogar es **pobre** o **no pobre** "
    "utilizando un clasificador Random Forest entrenado con la ENAHO 2024."
)
st.markdown("---")

with st.form("ENAHO_form"):
    col1, col2 = st.columns(2)

    with col1:
        dominio = st.selectbox(
            "**Dominio geográfico**",
            dominio_options,
            help="Región natural / ámbito de residencia del hogar"
        )
        mieperho = st.number_input(
            "**Total de miembros del hogar**",
            min_value=1, max_value=30, value=4, step=1
        )
        percepho = st.number_input(
            "**Número de perceptores de ingresos**",
            min_value=0, max_value=20, value=1, step=1,
            help="Personas dentro del hogar que generan algún ingreso"
        )

    with col2:
        estrato_label = st.selectbox(
            "**Estrato geográfico**",
            list(estrato_options.keys()),
            help="Tamaño del centro poblado según número de viviendas/habitantes"
        )
        gru71hd = st.number_input(
            "**Gasto en servicios de enseñanza (S/.)**",
            min_value=0.0, value=0.0, step=10.0,
            help="Gasto total del hogar en educación (grupo 71)"
        )

    col_pred, col_reset = st.columns([1, 1])
    with col_pred:
        predict_button = st.form_submit_button("🔍 Predecir", use_container_width=True)
    with col_reset:
        reset_button = st.form_submit_button("🔄 Limpiar", use_container_width=True)

# ─────────────────────────────────────────────
# 04. Predicción
# ─────────────────────────────────────────────
if predict_button:
    # Validaciones básicas
    if percepho > mieperho:
        st.error("⚠️ El número de perceptores no puede superar el total de miembros del hogar.")
        st.stop()

    # Recuperar el valor de estrato que el modelo conoce
    estrato_val = estrato_options[estrato_label]

    # Construir el DataFrame exactamente como lo espera el pipeline
    input_data = pd.DataFrame({
        'mieperho': [float(mieperho)],
        'percepho': [float(percepho)],
        'dominio':  [dominio],
        'estrato':  [estrato_val],
        'gru71hd':  [float(gru71hd)],
    })

    try:
        probabilities = clf.predict_proba(input_data)[0]
        class_predicted = int(np.argmax(probabilities))

        # Clase 0 = Pobre, Clase 1 = No pobre  (según notebook)
        if class_predicted == 0:
            outcome   = "🔴 Hogar POBRE"
            prob_show = probabilities[0]
            color     = "#f28b82"   # rojo suave
        else:
            outcome   = "🟢 Hogar NO POBRE"
            prob_show = probabilities[1]
            color     = "#81c995"   # verde suave

        st.markdown("### Resultado de la predicción")
        st.markdown(
            f"<div style='background-color:{color}; padding:16px; border-radius:8px; "
            f"font-size:1.1rem;'>"
            f"<b>{outcome}</b><br>"
            f"Probabilidad estimada: <b>{prob_show:.2%}</b>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Tabla resumen de entrada
        with st.expander("Ver datos ingresados"):
            st.dataframe(input_data.rename(columns={
                'mieperho': 'Miembros hogar',
                'percepho': 'Perceptores ingreso',
                'dominio':  'Dominio',
                'estrato':  'Estrato',
                'gru71hd':  'Gasto enseñanza (S/.)'
            }))

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")

if reset_button:
    st.rerun()

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Modelo entrenado con datos ENAHO 2024 | Random Forest Classifier")

# Para ejecutar:  streamlit run app_streamlit.py

"""Local Streamlit demonstration of a tabular CatBoost classifier."""
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

from preprocessing import PROJECT_DIR, encode_features


@st.cache_resource
def load_model():
    return CatBoostClassifier().load_model(
        str(PROJECT_DIR / "models" / "diabetes_prediction_model.cbm")
    )


st.set_page_config(page_title="Классификация табличных данных", page_icon="📊")
st.title("Учебный проект: классификация диабета")
st.caption("Демонстрация машинного обучения. Не предназначена для диагностики.")
try:
    model = load_model()
except Exception as error:
    st.error(f"Не удалось загрузить модель: {error}")
    st.stop()

with st.form("features"):
    gender_labels = {"Женский": "Female", "Мужской": "Male", "Другая категория": "Other"}
    gender = st.selectbox("Пол (категория датасета)", list(gender_labels))
    age = st.number_input("Возраст", min_value=0.0, max_value=120.0, value=25.0)
    hypertension = st.selectbox("Гипертония", ["Нет", "Да"])
    heart_disease = st.selectbox("Заболевания сердца", ["Нет", "Да"])
    smoking_labels = {
        "Нет информации": "No Info", "Курю": "current",
        "Курил когда-либо": "ever", "Бывший курильщик": "former",
        "Никогда не курил": "never", "Сейчас не курю": "not current",
    }
    smoking = st.selectbox("История курения", list(smoking_labels))
    bmi = st.number_input("ИМТ", min_value=1.0, max_value=100.0, value=22.5, step=0.1)
    hba1c = st.number_input("HbA1c", min_value=0.1, max_value=20.0, value=5.5, step=0.1)
    glucose = st.number_input("Глюкоза (мг/дл)", min_value=1, max_value=500, value=100)
    submitted = st.form_submit_button("Рассчитать")

if submitted:
    raw = pd.DataFrame([{
        "gender": gender_labels[gender], "age": age,
        "hypertension": int(hypertension == "Да"),
        "heart_disease": int(heart_disease == "Да"),
        "smoking_history": smoking_labels[smoking], "bmi": bmi,
        "HbA1c_level": hba1c, "blood_glucose_level": glucose,
    }])
    encoded = encode_features(raw)
    probability = float(model.predict_proba(encoded)[0, 1])
    st.dataframe(raw, hide_index=True)
    st.metric("Оценка модели для положительного класса", f"{probability:.1%}")
    st.caption("Выход модели не является подтверждённой вероятностью заболевания.")

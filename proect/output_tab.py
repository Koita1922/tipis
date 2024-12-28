import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
# Реализация приложения Streamlit
# Загрузка обученной модели
@st.cache_resource

def load_model():
    return CatBoostClassifier().load_model("diabetes_prediction_model.cbm")

# Загрузка модели
model = load_model()

# Интерфейс приложения
st.title("Приложение для предсказания диабета")
st.markdown(
    "Это приложение предсказывает вероятность диабета на основе введенных данных. Заполните поля ниже для получения предсказания."
)

# Поля ввода
def user_input_features():
    gender = st.selectbox("Пол", ["Мужской", "Женский"], index=0)
    age = st.number_input("Возраст", min_value=0, max_value=120, value=25, step=1)
    hypertension = st.selectbox("Гипертония (Высокое давление)", ["Нет", "Да"], index=0)
    heart_disease = st.selectbox("Заболевания сердца", ["Нет", "Да"], index=0)
    smoking_history = st.selectbox(
        "История курения", ["Никогда", "Бывший курильщик", "Курю", "Неизвестно"], index=0
    )
    bmi = st.number_input("ИМТ (Индекс массы тела)", min_value=0.0, max_value=50.0, value=22.5, step=0.1)
    HbA1c_level = st.number_input("Уровень HbA1c", min_value=0.0, max_value=20.0, value=5.5, step=0.1)
    blood_glucose_level = st.number_input(
        "Уровень глюкозы в крови (мг/дл)", min_value=0.0, max_value=500.0, value=100.0, step=1.0
    )

    # Преобразование категорий в числовые значения
    gender = 1 if gender == "Мужской" else 0
    hypertension = 1 if hypertension == "Да" else 0
    heart_disease = 1 if heart_disease == "Да" else 0

    smoking_mapping = {"Никогда": 0, "Бывший курильщик": 1, "Курю": 2, "Неизвестно": 3}
    smoking_history = smoking_mapping[smoking_history]

    return pd.DataFrame(
        {
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "smoking_history": [smoking_history],
            "bmi": [bmi],
            "HbA1c_level": [HbA1c_level],
            "blood_glucose_level": [blood_glucose_level],
        }
    )

# Получение данных от пользователя
input_df = user_input_features()

st.write("### Введенные данные")
st.write(input_df)

# Предсказание
if st.button("Предсказать"):  
    probability = model.predict_proba(input_df)[:, 1][0]  # Вероятность положительного класса
    st.write(f"### Вероятность наличия диабета: {probability * 100:.2f}%")
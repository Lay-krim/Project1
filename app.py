import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Функция для загрузки данных
@st.cache
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Функция для предобработки данных
@st.cache
def preprocess_data(data):
    data = pd.get_dummies(data, drop_first=True)
    scaler = StandardScaler()
    X = data.drop('charges', axis=1)
    y = data['charges']
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# Функция для загрузки модели
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Основная функция приложения
def main():
    st.title('Предсказание медицинских расходов')
    st.sidebar.title('Меню')

    # Загрузка пользовательских данных
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV файл с данными", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Загруженные данные:")
        st.write(data.head())
        
        if 'charges' not in data.columns:
            st.error("В данных должна быть колонка 'charges'")
            return
        
        # Обработка данных
        X, y, scaler = preprocess_data(data)
        
        # Визуализация данных
        st.sidebar.subheader('Визуализация данных')
        if st.sidebar.checkbox('Показать корреляционную матрицу'):
            st.subheader('Корреляционная матрица')
            corr_matrix = data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            st.pyplot(plt)
        
        # Обучение модели
        st.sidebar.subheader('Обучение модели')
        if st.sidebar.button('Обучить модель'):
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            
            # Сохранение модели
            model_path = 'ridge_model.pkl'
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            
            st.success('Модель обучена и сохранена')
            
        # Загрузка модели
        st.sidebar.subheader('Загрузка модели')
        if os.path.exists('ridge_model.pkl'):
            model = load_model('ridge_model.pkl')
            st.sidebar.success('Модель загружена')

            # Ввод данных пользователем
            st.sidebar.subheader('Ввод данных для предсказания')
            age = st.sidebar.slider('Возраст', 0, 100, 25)
            bmi = st.sidebar.slider('Индекс массы тела (BMI)', 0.0, 50.0, 25.0)
            children = st.sidebar.slider('Количество детей', 0, 10, 0)
            smoker = st.sidebar.selectbox('Курильщик', ['Да', 'Нет'])
            region = st.sidebar.selectbox('Регион', ['Юго-запад', 'Юго-восток', 'Северо-запад', 'Северо-восток'])

            smoker = 1 if smoker == 'Да' else 0
            region = [1 if region == r else 0 for r in ['Юго-запад', 'Юго-восток', 'Северо-запад', 'Северо-восток']]
            input_data = np.array([[age, bmi, children, smoker] + region])
            input_data = scaler.transform(input_data)

            if st.sidebar.button('Предсказать'):
                prediction = model.predict(input_data)
                st.sidebar.write(f'Предсказанные медицинские расходы: ${prediction[0]:.2f}')

if __name__ == '__main__':
    main()

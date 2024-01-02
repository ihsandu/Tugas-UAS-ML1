import streamlit as st
from functions import predict

def app(df, x, y):
    st.title("Halaman Prediksi")
    col1, col2 = st.columns(2)
    
    with col1:
        Pregnancies = st.text_input('Input Nilai Kehamilan')
    with col1:
        Glucose = st.text_input('Input Nilai Glukosa')
    with col1:
        BloodPressure = st.text_input('Input Nilai Tekanan Darah')
    with col1:
        SkinThickness = st.text_input('Input Nilai Ketebalan Kulit')
    with col2:
        Insulin = st.text_input('Input Nilai Insulin')
    with col2:
        BMI = st.text_input('Input Nilai BMI')
    with col2:
        DiabetesPedigreeFunction = st.text_input('Input Nilai Diabetes Silsilah Fungsi')
    with col2:
        Age = st.text_input('Input Nilai Usia')

    features = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]

    if st.button("Prediksi"):
        prediction, score = predict(x, y, features)
        score = score
        st.info("Prediksi Sukses...")

        if(prediction == 1):
           st.warning("Orang tersebut rentan terkena penyakit diabetes")
        else:
           st.success("Orang tersebut relaitf aman dari penyakit diabetes")

        st.write("Model yang digunakan memiliki tingkat akurasi ", (score*100),"%")
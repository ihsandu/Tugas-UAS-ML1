import warnings
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from functions import train_model

def app(df, x, y):

    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Visualisasi Prediksi Diabetes")

    if st.checkbox("Plot Confusion Matrix"):
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model, score = train_model(X_train, Y_train, X_test, Y_test)
        predictions = model.predict(X_test)
        cm = confusion_matrix(Y_test, predictions)
        plt.figure(figsize=(10,6))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        st.pyplot()

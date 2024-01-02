import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

@st.cache_data()
def load_data():
    df = pd.read_csv('diabetes.csv')
    X = df[["Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"]]
    y = df['Outcome']
    return df, X, y

@st.cache_data()
def train_model(X_train, y_train, X_test, y_test):
    test_scores = []
    train_scores = []
    
    for i in range(1, 15):
        knn = KNeighborsClassifier(i)
        knn.fit(X_train, y_train)
        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))

    best_k = test_scores.index(max(test_scores)) + 1
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    
    return model, score

def predict(X_train, y_train, X_test, y_test, features):
    model, score = train_model(X_train, y_train, X_test, y_test)

    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score

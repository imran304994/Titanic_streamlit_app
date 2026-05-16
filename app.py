import streamlit as st
import joblib
import numpy as np

model = joblib.load("titanic.pkl")

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex (0 = Male, 1 = Female)", [0, 1])
age = st.number_input("Age", min_value=0, max_value=100)
sibsp = st.number_input("Siblings / Spouses", min_value=0, max_value=10)
parch = st.number_input("Parents / Children", min_value=0, max_value=10)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

embarked_q = 0
embarked_s = 0

if embarked == "Q":
    embarked_q = 1
elif embarked == "S":
    embarked_s = 1

if st.button("Predict"):
    input_data = np.array([[pclass,sex,age,sibsp,parch,fare,embarked_q,embarked_s]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")

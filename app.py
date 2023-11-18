import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.svm import SVC

df = pd.read_csv(r"Salary_Data.csv")
df = df[["Age" , "Years of Experience", "Salary"]]
df = df[df["Salary"].notnull()]
df = df.dropna()

@st.cache_resource
def load_data():
    model = joblib.load("trained.pkl")
    return model

st.title("Salary Data Representation")
figure = plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),cmap="YlGnBu",annot=True)
st.header("Heat Map")
st.pyplot(figure)

st.header("Mean Salary based on Age")
data = df.groupby(["Age"])["Salary"].mean().sort_values(ascending=True)
st.bar_chart(data)

age = st.number_input("Enter age of candidate: ")
yrs = st.number_input("Enter years of experience: ")

btn = st.button("Predict")

if btn:
    model = load_data()
    st.subheader("Predicted Salary is: ")
    st.subheader(model.predict([[age,yrs]]))
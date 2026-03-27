import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("placement_model.pkl", "rb"))

st.title("  Placement Predictor")

st.write("Enter your details to predict placement chances")


cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
coding = st.slider("Coding Skills (1-10)", 1, 10, 5)
internships = st.number_input("Internships", 0, 5, 0)
communication = st.slider("Communication Skills (1-10)", 1, 10, 5)


if st.button("Predict"):
    features = np.array([[cgpa, coding, internships, communication]])
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    if prediction[0] == 1:
        st.success(f"✅ High chance of placement! ({probability[0][1]*100:.2f}%)")
    else:
        st.error(f"❌ Low chance of placement ({probability[0][1]*100:.2f}%)")
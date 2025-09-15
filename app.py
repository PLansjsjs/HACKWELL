import streamlit as st
import matplotlib.pyplot as plt
import shap
from wellness_model import WellnessAssistant

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("🩺 Multi-Disease AI Wellness Assistant")
st.write("Predict risk of **Diabetes, Heart Disease, and Hypertension** with personalized recommendations & explainability.")

assistant = WellnessAssistant()

# Diabetes inputs
st.sidebar.header("Diabetes Input Data")
diabetes_input = {
    "Pregnancies": st.sidebar.number_input("Pregnancies", 0, 20, 2),
    "Glucose": st.sidebar.number_input("Glucose Level", 50, 300, 120),
    "BloodPressure": st.sidebar.number_input("Blood Pressure", 50, 200, 80),
    "SkinThickness": st.sidebar.number_input("Skin Thickness", 0, 100, 20),
    "Insulin": st.sidebar.number_input("Insulin", 0, 500, 85),
    "BMI": st.sidebar.number_input("BMI", 10.0, 60.0, 25.0),
    "DiabetesPedigreeFunction": st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5),
    "Age": st.sidebar.number_input("Age (Diabetes)", 10, 100, 35),
}

# Heart inputs
st.sidebar.header("Heart Disease Input Data")
heart_input = {
    "age": st.sidebar.number_input("Age (Heart)", 10, 100, 40),
    "sex": st.sidebar.selectbox("Sex", [0, 1]),
    "cp": st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3]),
    "trestbps": st.sidebar.number_input("Resting BP", 80, 200, 120),
    "chol": st.sidebar.number_input("Cholesterol", 100, 600, 200),
    "fbs": st.sidebar.selectbox("Fasting Blood Sugar >120", [0, 1]),
    "restecg": st.sidebar.selectbox("Resting ECG", [0, 1, 2]),
    "thalach": st.sidebar.number_input("Max Heart Rate", 60, 220, 150),
    "exang": st.sidebar.selectbox("Exercise Angina", [0, 1]),
    "oldpeak": st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0),
    "slope": st.sidebar.selectbox("Slope of Peak Exercise", [0, 1, 2]),
    "ca": st.sidebar.selectbox("No. of Major Vessels", [0, 1, 2, 3]),
    "thal": st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3]),
}

# Hypertension inputs
st.sidebar.header("Hypertension Input Data")
hypertension_input = {
    "age": st.sidebar.number_input("Age (Hypertension)", 10, 100, 40),
    "sex": st.sidebar.selectbox("Sex (Hypertension)", [0, 1]),
    "BMI": st.sidebar.number_input("BMI (Hypertension)", 10.0, 60.0, 25.0),
    "systolicBP": st.sidebar.number_input("Systolic BP", 80, 200, 120),
    "diastolicBP": st.sidebar.number_input("Diastolic BP", 40, 120, 80),
    "cholesterol": st.sidebar.number_input("Cholesterol", 100, 600, 200),
    "smoking": st.sidebar.selectbox("Smoker", [0, 1]),
    "diabetes": st.sidebar.selectbox("Diabetes History", [0, 1]),
}

if st.sidebar.button("Predict Risks"):
    results = assistant.assess_health(diabetes_input, heart_input, hypertension_input)

    # Diabetes
    st.subheader("📊 Diabetes Risk")
    st.write(f"**Risk Probability:** {results['diabetes']['probability']:.2f}")
    for r in results["diabetes"]["recs"]:
        st.write("- " + r)

    shap_values, user_df = results["diabetes"]["shap"]
    st.subheader("🔍 Diabetes Explainability")
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[1][0],
        base_values=assistant.diabetes_model.explainer.expected_value[1],
        data=user_df.iloc[0],
        feature_names=assistant.diabetes_model.feature_names
    ))
    st.pyplot()

    # Heart
    st.subheader("❤️ Heart Disease Risk")
    st.write(f"**Risk Probability:** {results['heart']['probability']:.2f}")
    for r in results["heart"]["recs"]:
        st.write("- " + r)

    shap_values, user_df = results["heart"]["shap"]
    st.subheader("🔍 Heart Disease Explainability")
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[1][0],
        base_values=assistant.heart_model.explainer.expected_value[1],
        data=user_df.iloc[0],
        feature_names=assistant.heart_model.feature_names
    ))
    st.pyplot()

    # Hypertension
    st.subheader("🫀 Hypertension Risk")
    st.write(f"**Risk Probability:** {results['hypertension']['probability']:.2f}")
    for r in results["hypertension"]["recs"]:
        st.write("- " + r)

    shap_values, user_df = results["hypertension"]["shap"]
    st.subheader("🔍 Hypertension Explainability")
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[1][0],
        base_values=assistant.hypertension_model.explainer.expected_value[1],
        data=user_df.iloc[0],
        feature_names=assistant.hypertension_model.feature_names
    ))
    st.pyplot()

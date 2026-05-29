import streamlit as st
import pandas as pd 
import joblib as jb

model = jb.load('KNN_heart.pkl')
scaler = jb.load('scaler.pkl')
expected_columns = jb.load('columns.pkl')
st.title('💓 Heart Disease Prediction System')
st.markdown('This application predicts the likelihood of heart disease based on various health parameters. Please fill in the details below to get your prediction.')   

age = st.slider('Age', 18, 100, 40)
sex = st.selectbox('Sex', ['Male', 'Female'])
chest_pain = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
cholesterol = st.number_input('Serum Cholesterol (mg/dl)', 100, 600, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0,1])
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
max_heart_rate = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exercise_angina = st.selectbox('Exercise Induced Angina', [0,1])
oldpeak = st.slider('Oldpeak (ST depression)', 0.0, 10.0, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

if st.button("Predict"):
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex_M': [1 if sex == 'Male' else 0],
        'ChestPainType_ATA': [1 if chest_pain == 'Atypical Angina' else 0],
        'ChestPainType_NAP': [1 if chest_pain == 'Non-Anginal Pain' else 0],
        'ChestPainType_TA': [1 if chest_pain == 'Typical Angina' else 0],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG_Normal': [1 if resting_ecg == 'Normal' else 0],
        'RestingECG_ST': [1 if resting_ecg == 'ST-T Wave Abnormality' else 0],
        'MaxHR': [max_heart_rate],
        'ExerciseAngina_Y': [1 if exercise_angina == 1 else 0],
        'Oldpeak': [oldpeak],
        'ST_Slope_Flat': [1 if st_slope == 'Flat' else 0],
        'ST_Slope_Up': [1 if st_slope == 'Up' else 0]
    })
    input_df = input_data[expected_columns]
    input_df = input_df.fillna(0)
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
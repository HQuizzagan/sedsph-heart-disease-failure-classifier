import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from utilities import remappers

st.set_page_config(
    page_title="Predicting Heart Disease",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Predicting Heart Disease")

st.write("Enter the values of the 11 clinical features identified as highly predictive of the presence or absence of heart diseas.The trained Decision Tree model will predict whether you have heart disease or not.")

col1, col2, col3 = st.columns(3)

# Number of people
with col2:
    num_people = st.selectbox(
        label="How many people do you want to predict for?",
        options=[1, "Multiple"],
        index=0,
        help="Select the number of people you want to predict for.",
        key="num_people",
    )

if num_people == 1:
    st.sidebar.subheader('Clinical Features')
    st.sidebar.success("Please enter the following information about you to predict whether you have heart disease or not.")
    
    age = st.sidebar.slider(
        label="Age",
        min_value=18,
        max_value=100,
        value=50,
        step=1,
        key="age"
    )
    
    sex = st.sidebar.selectbox(
        label="Select your gender",
        options=["Male", "Female", "Other"],
        index=0,
        key='sex'
    )
    
    chest_pain_type = st.sidebar.selectbox(
        label="Chest Pain Type",
        options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
        index=0,
        key='chest_pain_type'
    )
    
    resting_bp = st.sidebar.slider(
        label="Resting Blood Pressure (mmHg)",
        min_value=0,
        max_value=200,
        value=120,
        step=1,
        key="resting_bp"
    )
    
    cholesterol = st.sidebar.slider(
        label="Cholesterol (mg/dl)",
        min_value=0,
        max_value=610,
        value=200,
        step=1,
        key="cholesterol"
    )
    
    fasting_bs = st.sidebar.selectbox(
        label="Fasting Blood Sugar > 120 mg/dl",
        options=[True, False],
        index=0,
        key='fasting_bs'
    )
    
    resting_ecg = st.sidebar.selectbox(
        label="Resting Electrocardiographic Results",
        options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
        index=0,
        key='resting_ecg'
    )
    
    max_hr = st.sidebar.slider(
        label="Maximum Heart Rate Achieved",
        min_value=60,
        max_value=220,
        value=120,
        step=1,
        key="max_hr"
    )
    
    exercise_angina = st.sidebar.selectbox(
        label="Exercise Induced Angina",
        options=["Yes", "No"],
        index=0,
        key='exercise_angina'
    )
    
    oldpeak = st.sidebar.slider(
        label="Oldpeak",
        min_value=-10.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        key="oldpeak"
    )
    
    st_slope = st.sidebar.selectbox(
        label="ST Slope",
        options=["Upsloping", "Flat", "Downsloping"],
        index=0,
        key='st_slope'
    )
    
    # Display the collected clinical variables as a table.
    feature_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    info_df = pd.DataFrame({
        "Clinical Features": feature_names,
        "Values": [age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]
    })
    
    # Remap the values for sex, chest_pain_type, resting_ecg, exercise_angina, st_slope using the remappers
    info_remapped_df = info_df.copy()
    to_remap = ['sex', 'chest_pain_type', 'resting_ecg', 'exercise_angina', 'st_slope']
    for col in to_remap:
        info_remapped_df['Values'] = info_remapped_df['Values'].replace(remappers.__dict__[col])
    
    st.markdown("---")
    
    st.subheader('Your Clinical Information')
    st.dataframe(
        data=info_df,
        # width=500,
        use_container_width=True,
        height=425
    )
    
    # Transpose the dataframe to make the "Clinical Features" column the column names
    info_remapped_df = info_remapped_df.T.reset_index(drop=True)
    info_remapped_df.columns = info_remapped_df.iloc[0]
    info_remapped_df = info_remapped_df.drop(0)
    # st.dataframe(info_remapped_df, use_container_width=True)
    
    colA, colB, colC = st.columns(3, gap="large")
    prediction = None
    with colB:
        if st.button('PREDICT HEART DISEASE!'):
            # Perform the prediction
            # Load the model
            model = load('models/decision_tree_classifier_pipeline.joblib')
            try:
                model_prediction = model.predict(info_remapped_df)
                prediction = model_prediction
                
                if prediction == 0:
                    st.balloons()
                else:
                    st.snow()
            except ValueError:
                st.error("Please fill in all the information above.")
            except Exception as e:
                st.error(
                    f"Oops! Something went wrong. Please take a screenshot of this page and raise an issue on the Streamlit GitHub page.\n\n{e}"
                )
                
    if prediction == 0:
        st.success("Congratulations! You don't have heart disease.")
    else:
        st.error("Oops! You have heart disease.")
    
else:
    st.info('Download the following CSV file and fill in the necessary information for the people you want to predict for.')
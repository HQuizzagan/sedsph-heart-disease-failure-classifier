import streamlit as st

st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="♥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Heart Disease Classifier")

st.markdown(
    '''
    This app utilizes a **pre-trained** machine learning model. The ML model is a `DecisionTreeClassifier` that used 80% of the data to train and 20% to test. The model was trained on the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) dataset from Kaggle. Using 11 *clinical features* which is a combination of *numerical* and *categorical* data, the classifier can predict whether the corresponding person has a heart disease or not.
    
    The model was trained to predict whether a person has heart disease or not based on the input parameters. Based on the performance of the model, it is able to predict with an approximately 84% accuracy.
    
    **LIMITATIONS:**
    The model can classify whether the person has heart disease or not, but it can NOT identify what type of heart disease the person has. Further, the dataset it was trained on is not a very large dataset, so the model may not be able to generalize well to other datasets.
    '''
)

st.info('Navigate through the pages of the app to view different sections relevant to you!')
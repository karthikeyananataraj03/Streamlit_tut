
import xgboost as xgb
import streamlit as st
import pandas as pd

#Loading up the Regression model we created
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

#Caching the model for faster loading
@st.cache


def predict(hospitalized,hospitalizedCurrently,hospitalizedIncrease,negativeIncrease,positiveIncrease,totalTestResultsIncrease,recovered):
    #Predicting the price of the carat
    
    

    prediction = model.predict(pd.DataFrame([[hospitalized,hospitalizedCurrently,hospitalizedIncrease,negativeIncrease,positiveIncrease,totalTestResultsIncrease,recovered]],
                                           columns=['hospitalized','hospitalizedCurrently','hospitalizedIncrease','positiveIncrease','totalTestResultsIncrease','recovered']))
    return prediction



st.title('Covid death count predictor')

st.header('Enter the Details:')
hospitalized = st.number_input('hospitalized:', min_value=0.1, max_value=100000000.0, value=1.0)

hospitalizedCurrently = st.number_input('hospitalizedCurrently:', min_value=0.1, max_value=100000000.0, value=1.0)

hospitalizedIncrease = st.number_input('hospitalizedIncrease:', min_value=0.1, max_value=100000000.0, value=1.0)
positiveIncrease = st.number_input('positiveIncrease:', min_value=0.1, max_value=100000000.0, value=1.0)
totalTestResultsIncrease = st.number_input('totalTestResultsIncrease:', min_value=0.1, max_value=100000000.0, value=1.0)
recovered = st.number_input('recovered:', min_value=0.1, max_value=100000000.0, value=1.0)


if st.button('Predict Death'):
    Death = predict(hospitalized,hospitalizedCurrently,hospitalizedIncrease,negativeIncrease,positiveIncrease,totalTestResultsIncrease,recovered)
    st.success(f'The predicted deaths  ${Death[0]:.2f} people')

import xgboost as xgb
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv("Stremlit_data.csv")
df = df.sort_values('date', ascending=False)

st.title('Covid death count predictor')
st.line_chart(data=df, x='date', y=['deathIncrease', 'hospitalizedIncrease', 'inIcuCurrently'], width=0, height=0, use_container_width=True)


#Loading up the Regression model we created

model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

#Caching the model for faster loading
@st.cache


def predict(hospitalized,hospitalizedCurrently,hospitalizedIncrease,negativeIncrease,positiveIncrease,totalTestResultsIncrease,recovered):
    #Predicting the price of the carat
    
    

    prediction = model.predict(pd.DataFrame([[hospitalized,hospitalizedCurrently,hospitalizedIncrease,negativeIncrease,positiveIncrease,totalTestResultsIncrease,recovered]],
                                           columns=['hospitalized','hospitalizedCurrently','hospitalizedIncrease','negativeIncrease','positiveIncrease','totalTestResultsIncrease','recovered']))
    return prediction





st.header('Enter the Details:')

hospitalized = st.number_input('hospitalized:')
hospitalizedCurrently = st.number_input('hospitalizedCurrently:')
hospitalizedIncrease = st.number_input('hospitalizedIncrease:')
negativeIncrease = st.number_input('negativeIncrease:')
positiveIncrease = st.number_input('positiveIncrease:')
totalTestResultsIncrease = st.number_input('totalTestResultsIncrease:')
recovered = st.number_input('recovered:')


if st.button('Predict Death'):
    Death = predict(hospitalized,hospitalizedCurrently,hospitalizedIncrease,negativeIncrease,positiveIncrease,totalTestResultsIncrease,recovered)
    st.success(f'The predicted deaths  ${Death[0]:.2f} people')

#Gender 1 female 0 male 
#Churn 1 yes 0 no
#scaler is exported as scaler.pkl
#Model is exported as model.pkl
#Order of X is : 'Age', 'Gender', 'Tenure', 'MonthlyCharges'
import streamlit as st
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))


st.title("Churn Prediction App")
st.divider()
st.write("Enter the details of the customer to predict whether the customer will churn or not.")
st.divider()

age=st.number_input("Enter Age",min_value=18,max_value=100,value=30)
tenure=st.number_input("Enter the Tenure (in months)",min_value=0,max_value=130,value=10)
monthlycharges=st.number_input("Enter the Monthly Charges",min_value=30,max_value=150)
gender=st.selectbox("Enter the Gender",["Male",'Female'])
st.divider()

if gender:
    gender_selected=1 if gender =='Female' else 0

predictbutton=st.button("Predict!")
if predictbutton:
     gender_selected=1 if gender =='Female' else 0
     X=[age,gender_selected,tenure,monthlycharges]
     X1=np.array(X)
     X_scaled=scaler.transform([X1])
     prediction=model.predict(X_scaled)
     st.divider()
     if prediction[0]==1:
         st.write("The customer is likely to churn.")
     else:
         st.write("The customer is not likely to churn.")
else:
    st.write("Please enter all the details and click on Predict")

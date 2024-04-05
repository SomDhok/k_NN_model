import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# Load the dataset (assuming the CSV is in the same directory)
data = pd.read_csv("Social_Network_Ads.csv")

# Define features and target variable
features = ["Age", "EstimatedSalary"]
target = "Purchased"

# Normalize features (performed once at startup)
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
normalized_features = data[features]

# Create the KNN model 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(normalized_features, data[target])

# Title 
st.title("Car Purchase Prediction with KNN")
st.header("**Input User Information**")

# User input
age = st.number_input("Enter your age:", min_value=18, max_value=100)
salary = st.number_input("Enter your estimated salary:", min_value=0)

# Normalize user input (performed on each prediction)
new_data = pd.DataFrame([[age, salary]], columns=features)
new_data_normalized = scaler.transform(new_data)

# Prediction button
if st.button("Predict"):
  prediction = knn.predict(new_data_normalized)
  
  # Display prediction
  if prediction[0] == 0:
    st.success("Prediction: You are unlikely to purchase a car.")
  else:
    st.success("Prediction: You are likely to purchase a car.")


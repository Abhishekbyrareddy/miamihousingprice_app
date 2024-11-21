import streamlit as st
import numpy as np
import os
import joblib

def main():
    st.title("Miami Housing Price Prediction App")

    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), 'miami_housing_price_model.joblib')
    model = joblib.load(model_path)

    # User inputs for features
    land_size = st.number_input("Enter the land area (sq. ft):", min_value=500, max_value=100000, value=7500)
    living_area = st.number_input("Enter the total living area (sq. ft):", min_value=500, max_value=10000, value=2000)
    property_age = st.slider("Enter the property age (years):", 0, 100, 30)
    structure_quality = st.selectbox("Enter the structure quality:", [1, 2, 3, 4, 5])
    rail_dist = st.number_input("Distance to the nearest rail station (miles):", value=5.0)
    ocean_dist = st.number_input("Distance to the ocean (miles):", value=10.0)

    # Prepare input array for prediction
    input_data = np.array([[land_size, living_area, property_age, structure_quality, rail_dist, ocean_dist]])

    # Predict and display the result
    if st.button("Predict"):
        try:
            prediction = model.predict(input_data)
            st.success(f"The predicted house price is ${round(prediction[0], 2)}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == '__main__':
    main()

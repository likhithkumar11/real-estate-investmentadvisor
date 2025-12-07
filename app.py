import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

clf = joblib.load(os.path.join(BASE, "investment_classifier.pkl"))
reg = joblib.load(os.path.join(BASE, "price_regressor.pkl"))

st.title("🏠 Real Estate Investment Advisor")

city = st.text_input("City", value="Unknown")
ptype = st.text_input("Property Type", value="Apartment")
bhk = st.number_input("BHK", min_value=1, value=2)
size = st.number_input("Size in SqFt", min_value=100, value=1000)
price = st.number_input("Current Price in Lakhs", min_value=1.0, value=50.0)
school = st.slider("Nearby Schools", 0, 10, 3)
hospital = st.slider("Nearby Hospitals", 0, 10, 2)
parking = st.selectbox("Parking Available", [0,1], index=1)
furnish = st.selectbox("Furnished (Yes=1, No=0)", [0,1], index=1)

if st.button("Predict Investment & Future Price"):

    input_df = pd.DataFrame([[
        int(bhk),
        float(size),
        float(price),
        int(school),
        int(hospital),
        str(city).strip(),
        str(ptype).strip(),
        str(parking),
        str(furnish)
    ]], columns=[
        'BHK','Size_in_SqFt','Price_in_Lakhs',
        'Nearby_Schools','Nearby_Hospitals',
        'City','Property_Type','Parking_Space','Furnished_Status'
    ])

    # FORCE CLEAN
    for col in ['City','Property_Type','Parking_Space','Furnished_Status']:
        input_df[col] = (
            input_df[col]
            .astype(str)
            .replace(['nan','NaN','None',''], 'Unknown')
        )

    try:
        invest = clf.predict(input_df)[0]
        prob = clf.predict_proba(input_df)[0][1]
        future = reg.predict(input_df)[0]

        st.success("Prediction Complete ✅")
        st.write("✅ Good Investment" if invest==1 else "❌ Not a Good Investment")
        st.write(f"Confidence Score: {prob:.2f}")
        st.write(f"Estimated Price After 5 Years: ₹ {future:.2f} Lakhs")

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))

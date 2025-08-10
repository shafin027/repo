# app.py - Fixed Flight Price Predictor
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Configuration
# -------------------------------
DATA_FILE = 'cleaned_flight_data.csv'  # Must run fix_data.py first!
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, 'flight_model.pkl')
encoder_path = os.path.join(MODEL_DIR, 'encoders.pkl')
duration_path = os.path.join(MODEL_DIR, 'durations.pkl')
price_trend_path = os.path.join(MODEL_DIR, 'price_trends.pkl')

# -------------------------------
# Load & Clean Data
# -------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ File '{DATA_FILE}' not found. Please run 'fix_data.py' first.")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    # Convert dates
    df['Departure'] = pd.to_datetime(df['Departure'], errors='coerce')
    df['Arrival'] = pd.to_datetime(df['Arrival'], errors='coerce')

    # Extract time features
    df['DepartureDay'] = df['Departure'].dt.day
    df['DepartureMonth'] = df['Departure'].dt.month
    df['DepartureHour'] = df['Departure'].dt.hour
    df['Weekday'] = df['Departure'].dt.weekday

    # Clean TotalPrice
    df['TotalPrice'] = pd.to_numeric(df['TotalPrice'], errors='coerce')
    df.dropna(subset=['TotalPrice'], inplace=True)

    # Drop unnecessary
    df.drop(['SourceName', 'DestinationName', 'FlightID', 'BasePrice', 'Tax'], axis=1, inplace=True)

    return df

# -------------------------------
# Train or Load Model
# -------------------------------
if os.path.exists(model_path):
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoder_path)
    route_durations = joblib.load(duration_path)
    route_avg_prices = joblib.load(price_trend_path)
    st.sidebar.success("✅ Model loaded!")
else:
    st.sidebar.info("🔧 Training model...")

    df = load_data()

    # Route-based duration
    route_durations = df.groupby(['Source', 'Destination'])['Duration'].mean().to_dict()
    route_avg_prices = df.groupby(['Source', 'Destination'])['TotalPrice'].mean().round(2).to_dict()

    # Encode categorical columns
    cat_cols = ['Airline', 'Source', 'Destination', 'Stops', 'Aircraft', 'Class', 'BookingMethod', 'Season']
    label_encoders = {}
    df_encoded = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    # Features
    feature_cols = ['Airline', 'Source', 'Destination', 'Stops', 'Aircraft', 'Class',
                    'BookingMethod', 'Season', 'DepartureDay', 'DepartureMonth',
                    'DepartureHour', 'Weekday', 'Duration']
    X = df_encoded[feature_cols]
    y = df_encoded['TotalPrice']

    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save
    joblib.dump(model, model_path)
    joblib.dump(label_encoders, encoder_path)
    joblib.dump(route_durations, duration_path)
    joblib.dump(route_avg_prices, price_trend_path)

    st.sidebar.success("✅ Model trained and saved!")

# -------------------------------
# Helper
# -------------------------------
def get_categories(col):
    return label_encoders[col].classes_

# -------------------------------
# UI
# -------------------------------
st.title("✈️ Bangladesh Flight Price Predictor")

airline = st.selectbox("Airline", get_categories('Airline'))
source = st.selectbox("Source", get_categories('Source'))
destination = st.selectbox("Destination", get_categories('Destination'))

if source == destination:
    st.error("❌ Source and destination cannot be the same.")
else:
    default_duration = route_durations.get((source, destination), 1.0)
    duration = st.number_input("Duration (hours)", value=round(default_duration, 2))

    stops = st.selectbox("Stops", get_categories('Stops'))
    aircraft = st.selectbox("Aircraft", get_categories('Aircraft'))
    flight_class = st.selectbox("Class", get_categories('Class'))
    booking_method = st.selectbox("Booking Method", get_categories('BookingMethod'))
    season = st.selectbox("Season", get_categories('Season'))

    date = st.date_input("Departure Date")
    time = st.time_input("Departure Time")

    departure_day = date.day
    departure_month = date.month
    departure_hour = time.hour
    weekday = date.weekday()

    if st.button("🔍 Predict Price"):
        try:
            input_df = pd.DataFrame([[
                airline, source, destination, stops, aircraft, flight_class,
                booking_method, season, departure_day, departure_month,
                departure_hour, weekday, duration
            ]], columns=[
                'Airline', 'Source', 'Destination', 'Stops', 'Aircraft', 'Class',
                'BookingMethod', 'Season', 'DepartureDay', 'DepartureMonth',
                'DepartureHour', 'Weekday', 'Duration'
            ])

            for col in ['Airline', 'Source', 'Destination', 'Stops', 'Aircraft', 'Class', 'BookingMethod', 'Season']:
                input_df[col] = label_encoders[col].transform([input_df[col].iloc[0]])

            price = model.predict(input_df)[0]
            st.success(f"💰 Predicted Price: BDT {price:,.2f}")

            avg_price = route_avg_prices.get((source, destination), "N/A")
            if avg_price != "N/A":
                st.info(f"📊 Average historical price for {source} → {destination}: BDT {avg_price:,.2f}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

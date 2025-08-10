# app.py - Enhanced Flight Price Predictor for Bangladesh
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# -------------------------------
# 1. Load or Train Model & Save with Joblib
# -------------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "flight_price_model.pkl")
encoder_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
duration_dict_path = os.path.join(MODEL_DIR, "route_durations.pkl")
price_trend_path = os.path.join(MODEL_DIR, "route_avg_prices.pkl")

@st.cache_data
def load_or_train_model():
    df_raw = pd.read_csv('Flight_Price_Dataset_of_Bangladesh.csv', header=None)
    columns = [
        'Airline', 'Source', 'SourceName', 'Destination', 'DestinationName',
        'Departure', 'Arrival', 'Duration', 'Stops', 'Aircraft', 'Class',
        'BookingMethod', 'BasePrice', 'Tax', 'TotalPrice', 'Season', 'FlightID'
    ]
    df_raw.columns = columns

    # Convert datetime
    df_raw['Departure'] = pd.to_datetime(df_raw['Departure'])
    df_raw['Arrival'] = pd.to_datetime(df_raw['Arrival'])

    # Extract features
    df_raw['DepartureDay'] = df_raw['Departure'].dt.day
    df_raw['DepartureMonth'] = df_raw['Departure'].dt.month
    df_raw['DepartureHour'] = df_raw['Departure'].dt.hour
    df_raw['Weekday'] = df_raw['Departure'].dt.weekday

    # Clean
    df_raw['TotalPrice'] = pd.to_numeric(df_raw['TotalPrice'], errors='coerce')
    df = df_raw.dropna().copy()

    # Drop unnecessary
    df.drop(['SourceName', 'DestinationName', 'Departure', 'Arrival', 'FlightID', 'BasePrice', 'Tax'], axis=1, inplace=True)

    # -------------------------------
    # Create Route-to-Duration Dictionary
    # -------------------------------
    route_durations = df.groupby(['Source', 'Destination'])['Duration'].mean().to_dict()

    # -------------------------------
    # Create Price Trends: Avg Price by Route
    # -------------------------------
    route_avg_prices = df.groupby(['Source', 'Destination'])['TotalPrice'].mean().round(2).to_dict()

    # Encode categorical variables
    categorical_cols = ['Airline', 'Source', 'Destination', 'Stops', 'Aircraft', 'Class', 'BookingMethod', 'Season']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Features
    X = df.drop('TotalPrice', axis=1)
    y = df['TotalPrice']

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save everything
    joblib.dump(model, model_path)
    joblib.dump(label_encoders, encoder_path)
    joblib.dump(route_durations, duration_dict_path)
    joblib.dump(route_avg_prices, price_trend_path)

    return model, label_encoders, route_durations, route_avg_prices

# Try to load pre-trained model, otherwise train and save
if os.path.exists(model_path):
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoder_path)
    route_durations = joblib.load(duration_dict_path)
    route_avg_prices = joblib.load(price_trend_path)
    st.session_state['model_loaded'] = True
else:
    model, label_encoders, route_durations, route_avg_prices = load_or_train_model()
    st.success("✅ Model trained and saved for future use!")

# -------------------------------
# 2. Streamlit UI
# -------------------------------
st.title("✈️ Bangladesh Flight Price Predictor")
st.markdown("_Predict prices & compare routes intelligently._")

# Get original categories
def get_categories(col):
    return label_encoders[col].classes_

# Input Form
st.sidebar.header("🛫 Enter Flight Details")

airline = st.sidebar.selectbox("Airline", get_categories('Airline'))
source = st.sidebar.selectbox("Source", get_categories('Source'))
destination = st.sidebar.selectbox("Destination", get_categories('Destination'))

if source == destination:
    st.sidebar.error("❌ Source and Destination cannot be the same.")
    st.stop()

# Auto-fill duration based on route
default_duration = route_durations.get((source, destination), 1.0)
duration = st.sidebar.number_input("Duration (hours)", value=round(default_duration, 2), min_value=0.1, step=0.1)

stops = st.sidebar.selectbox("Stops", get_categories('Stops'))
aircraft = st.sidebar.selectbox("Aircraft", get_categories('Aircraft'))
flight_class = st.sidebar.selectbox("Class", get_categories('Class'))
booking_method = st.sidebar.selectbox("Booking Method", get_categories('BookingMethod'))
season = st.sidebar.selectbox("Season", get_categories('Season'))

# Date & Time
date = st.sidebar.date_input("Departure Date", value=datetime.today())
time = st.sidebar.time_input("Departure Time", value=datetime.now().time())

# Extract features
departure_day = date.day
departure_month = date.month
departure_hour = time.hour
weekday = date.weekday()

# Input DataFrame
input_data = pd.DataFrame([[
    airline, source, destination, stops, aircraft, flight_class,
    booking_method, season, departure_day, departure_month,
    departure_hour, weekday, duration
]], columns=[
    'Airline', 'Source', 'Destination', 'Stops', 'Aircraft', 'Class',
    'BookingMethod', 'Season', 'DepartureDay', 'DepartureMonth',
    'DepartureHour', 'Weekday', 'Duration'
])

# Encode inputs
for col in ['Airline', 'Source', 'Destination', 'Stops', 'Aircraft', 'Class', 'BookingMethod', 'Season']:
    if input_data[col].iloc[0] not in label_encoders[col].classes_:
        st.error(f"⚠️ Unknown value: {col} = {input_data[col].iloc[0]}")
        st.stop()
    input_data[col] = label_encoders[col].transform([input_data[col].iloc[0]])

# Predict Button
if st.sidebar.button("🔍 Predict Price"):
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"💰 **Predicted Price: BDT {predicted_price:,.2f}**")
    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# 3. Show Price Trend for Route
# -------------------------------
st.subheader("📊 Historical Price Trend")
route_key = (source, destination)
if route_key in route_avg_prices:
    avg_price = route_avg_prices[route_key]
    st.info(f"📌 Average historical price for **{source} → {destination}**: **BDT {avg_price:,.2f}**")
else:
    st.info("No historical data available for this route.")

# Optional: Show feature importance
if st.checkbox("📈 Show Feature Importance"):
    importance_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    st.bar_chart(importance_df.set_index('Feature'))

# Footer
st.markdown("---")
st.markdown("💡 *Trained on real flight data from Bangladesh. Perfect for comparing airlines, seasons, and booking strategies.*")

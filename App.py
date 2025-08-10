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
    # Load the dataset with header
    df_raw = pd.read_csv('Flight_Price_Dataset_of_Bangladesh.csv')

    # Convert datetime columns (using correct names)
    df_raw['Departure Date & Time'] = pd.to_datetime(df_raw['Departure Date & Time'], errors='coerce')
    df_raw['Arrival Date & Time'] = pd.to_datetime(df_raw['Arrival Date & Time'], errors='coerce')

    # Extract features from correct datetime columns
    df_raw['DepartureDay'] = df_raw['Departure Date & Time'].dt.day
    df_raw['DepartureMonth'] = df_raw['Departure Date & Time'].dt.month
    df_raw['DepartureHour'] = df_raw['Departure Date & Time'].dt.hour
    df_raw['Weekday'] = df_raw['Departure Date & Time'].dt.weekday

    # Clean Total Fare (BDT) and drop rows with NaNs created by datetime conversion errors
    df_raw['Total Fare (BDT)'] = pd.to_numeric(df_raw['Total Fare (BDT)'], errors='coerce')
    df = df_raw.dropna(subset=['Total Fare (BDT)', 'Departure Date & Time', 'Arrival Date & Time']).copy()

    # Drop unnecessary columns (using correct names)
    df.drop(['Source Name', 'Destination Name', 'Departure Date & Time', 'Arrival Date & Time', 'Base Fare (BDT)', 'Tax & Surcharge (BDT)'], axis=1, inplace=True)

    # -------------------------------
    # Create Route-to-Duration Dictionary
    # -------------------------------
    # Ensure 'Duration (hrs)' is numeric
    df['Duration (hrs)'] = pd.to_numeric(df['Duration (hrs)'], errors='coerce')
    df.dropna(subset=['Duration (hrs)'], inplace=True)
    route_durations = df.groupby(['Source', 'Destination'])['Duration (hrs)'].mean().to_dict()


    # -------------------------------
    # Create Price Trends: Avg Price by Route
    # -------------------------------
    route_avg_prices = df.groupby(['Source', 'Destination'])['Total Fare (BDT)'].mean().round(2).to_dict()

    # Encode categorical variables (using correct column names)
    categorical_cols = ['Airline', 'Source', 'Destination', 'Stopovers', 'Aircraft Type', 'Class', 'Booking Source', 'Seasonality']
    label_encoders = {}
    for col in categorical_cols:
        # Ensure column exists before encoding
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
             print(f"Warning: Column '{col}' not found in DataFrame for encoding.")


    # Features
    X = df.drop('Total Fare (BDT)', axis=1)
    y = df['Total Fare (BDT)']

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
    try:
        model = joblib.load(model_path)
        label_encoders = joblib.load(encoder_path)
        route_durations = joblib.load(duration_dict_path)
        route_avg_prices = joblib.load(price_trend_path)
        st.session_state['model_loaded'] = True
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        st.info("Retrying to train the model...")
        model, label_encoders, route_durations, route_avg_prices = load_or_train_model()
        st.success("✅ Model trained and saved for future use!")
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
    if col in label_encoders:
        return label_encoders[col].classes_
    else:
        return ["Unknown"] # Or handle missing columns appropriately

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

stops = st.sidebar.selectbox("Stops", get_categories('Stopovers')) # Use correct column name
aircraft = st.sidebar.selectbox("Aircraft", get_categories('Aircraft Type')) # Use correct column name
flight_class = st.sidebar.selectbox("Class", get_categories('Class'))
booking_method = st.sidebar.selectbox("Booking Method", get_categories('Booking Source')) # Use correct column name
season = st.sidebar.selectbox("Season", get_categories('Seasonality')) # Use correct column name

# Date & Time
date = st.sidebar.date_input("Departure Date", value=datetime.today())
time = st.sidebar.time_input("Departure Time", value=datetime.now().time())

# Extract features
departure_day = date.day
departure_month = date.month
departure_hour = time.hour
weekday = date.weekday()

# Input DataFrame (using correct column names)
input_data = pd.DataFrame([[
    airline, source, destination, stops, aircraft, flight_class,
    booking_method, season, departure_day, departure_month,
    departure_hour, weekday, duration
]], columns=[
    'Airline', 'Source', 'Destination', 'Stopovers', 'Aircraft Type', 'Class',
    'Booking Source', 'Seasonality', 'DepartureDay', 'DepartureMonth',
    'DepartureHour', 'Weekday', 'Duration (hrs)' # Use correct column name 'Duration (hrs)'
])

# Encode inputs (using correct column names)
categorical_input_cols = ['Airline', 'Source', 'Destination', 'Stopovers', 'Aircraft Type', 'Class', 'Booking Source', 'Seasonality']
for col in categorical_input_cols:
    if col in label_encoders: # Check if encoder exists for the column
        if input_data[col].iloc[0] not in label_encoders[col].classes_:
             st.error(f"⚠️ Unknown value for {col}: {input_data[col].iloc[0]}. Please select a value from the available options.")
             st.stop()
        input_data[col] = label_encoders[col].transform([input_data[col].iloc[0]])
    else:
        st.error(f"Encoder not found for column: {col}. Cannot process input.")
        st.stop()


# Predict Button
if st.sidebar.button("🔍 Predict Price"):
    try:
        # Ensure numerical columns have correct dtype before prediction
        numerical_input_cols = ['DepartureDay', 'DepartureMonth', 'DepartureHour', 'Weekday', 'Duration (hrs)']
        for col in numerical_input_cols:
             if col in input_data.columns:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
             else:
                 st.error(f"Missing numerical input column: {col}")
                 st.stop()

        # Drop any columns in input_data that were dropped during training but are still present
        # This is a safety measure
        cols_to_drop_from_input = [col for col in input_data.columns if col not in model.feature_names_in_]
        if cols_to_drop_from_input:
            input_data.drop(columns=cols_to_drop_from_input, inplace=True)

        # Reorder input columns to match the order the model was trained on
        input_data = input_data[model.feature_names_in_]

        predicted_price = model.predict(input_data)[0]
        st.success(f"💰 **Predicted Price: BDT {predicted_price:,.2f}**")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        # Optional: Print the input_data DataFrame to the console for debugging
        # print("Input data before prediction:")
        # print(input_data)

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
# Check if model has feature_importances_ attribute (RandomForestRegressor does)
if hasattr(model, 'feature_importances_') and st.checkbox("📈 Show Feature Importance"):
    # Need to get the feature names after one-hot encoding from the original training process
    # This is complex because the trained model inside joblib doesn't store the preprocessor's output feature names easily.
    # A simpler approach for this streamlit app is to get feature importances from the trained model directly,
    # but mapping them back to original categorical features after one-hot encoding is tricky without re-applying
    # the preprocessor and getting the feature names from the OneHotEncoder.
    # For now, showing the feature importances based on the features the model was trained on (post-encoding, pre-scaling if done)
    # This requires knowing the order and names of features the model received during training.
    # Since we trained the RandomForestRegressor directly on the DataFrame after one-hot encoding and dropping columns,
    # the feature names should correspond to the columns of that DataFrame.

    # Recreate the columns of the DataFrame that the model was trained on
    # This assumes the order and columns are consistent.
    # A more robust way would be to save the preprocessor and get feature names from it.
    # For this example, we rely on model.feature_names_in_ if available (sklearn >= 1.0)

    if hasattr(model, 'feature_names_in_'):
         feature_names = model.feature_names_in_
         importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
         st.bar_chart(importance_df.set_index('Feature'))
    else:
        st.warning("Feature importance display is not available for this model version or type.")


# Footer
st.markdown("---")
st.markdown("💡 *Trained on real flight data from Bangladesh. Perfect for comparing airlines, seasons, and booking strategies.*")
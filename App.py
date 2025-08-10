import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
import uuid

# Function to load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv('Flight_Price_Dataset_of_Bangladesh.csv')
    
    airport_mappings = df[['Source', 'Source Name', 'Destination', 'Destination Name']].drop_duplicates()
    source_map = dict(zip(airport_mappings['Source Name'], airport_mappings['Source']))
    destination_map = dict(zip(airport_mappings['Destination Name'], airport_mappings['Destination']))
    
    features = ['Airline', 'Source', 'Destination', 'Duration (hrs)', 'Stopovers', 
                'Aircraft Type', 'Class', 'Booking Source', 'Seasonality', 'Days Before Departure']
    target = 'Total Fare (BDT)'
    
    label_encoders = {}
    for col in ['Airline', 'Source', 'Destination', 'Stopovers', 'Aircraft Type', 
                'Class', 'Booking Source', 'Seasonality']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    X = df[features]
    y = df[target]
    
    return X, y, label_encoders, df, source_map, destination_map

# Function to train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to predict fare
def predict_fare(model, label_encoders, feature_columns, input_data):
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    
    for col in ['Airline', 'Source', 'Destination', 'Stopovers', 'Aircraft Type', 
                'Class', 'Booking Source', 'Seasonality']:
        try:
            input_df[col] = label_encoders[col].transform([input_data[col]])[0]
        except ValueError:
            st.error(f"Invalid value for {col}. Please select a valid option from the dropdown.")
            return None
    
    return model.predict(input_df)[0]

# Streamlit app
def main():
    # Set page configuration
    st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")
    
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stNumberInput {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #1a3c6e;
        font-family: 'Arial', sans-serif;
    }
    .stSuccess {
        background-color: #e6f3e6;
        border: 1px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("✈️ Flight Price Predictor")
    st.markdown("Select your flight details to predict the total fare in BDT with our intuitive and user-friendly interface.")

    # Load and preprocess data
    X, y, label_encoders, df, source_map, destination_map = load_and_preprocess_data()
    
    # Train the model
    model = train_model(X, y)
    
    # Create input form with improved layout
    with st.form("flight_form"):
        st.subheader("Flight Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            airline = st.selectbox("Airline", options=label_encoders['Airline'].classes_, help="Choose your preferred airline")
            source_name = st.selectbox("Source Airport", options=sorted(source_map.keys()), help="Select departure airport")
            destination_name = st.selectbox("Destination Airport", options=sorted(destination_map.keys()), help="Select arrival airport")
            stopovers = st.selectbox("Stopovers", options=label_encoders['Stopovers'].classes_, help="Number of stopovers")
            aircraft_type = st.selectbox("Aircraft Type", options=label_encoders['Aircraft Type'].classes_, help="Type of aircraft")
        
        with col2:
            class_type = st.selectbox("Class", options=label_encoders['Class'].classes_, help="Select travel class")
            booking_source = st.selectbox("Booking Source", options=label_encoders['Booking Source'].classes_, help="Where are you booking from?")
            seasonality = st.selectbox("Seasonality", options=label_encoders['Seasonality'].classes_, help="Travel season")
            duration = st.number_input("Duration (hours)", min_value=0.0, max_value=24.0, value=1.0, step=0.1, help="Flight duration in hours")
            days_before = st.number_input("Days Before Departure", min_value=0, max_value=365, value=30, step=1, help="Days until departure")
        
        budget = st.number_input("Your Budget (BDT)", min_value=0.0, value=30000.0, step=1000.0, help="Your maximum budget in BDT")
        
        # Submit button
        submitted = st.form_submit_button("Predict Fare")
    
    if submitted:
        input_data = {
            'Airline': airline,
            'Source': source_map[source_name],
            'Destination': destination_map[destination_name],
            'Duration (hrs)': duration,
            'Stopovers': stopovers,
            'Aircraft Type': aircraft_type,
            'Class': class_type,
            'Booking Source': booking_source,
            'Seasonality': seasonality,
            'Days Before Departure': days_before
        }
        
        predicted_fare = predict_fare(model, label_encoders, X.columns, input_data)
        
        if predicted_fare is not None:
            st.success(f"**Predicted Total Fare: {predicted_fare:.2f} BDT**")
            
            if predicted_fare <= budget:
                st.markdown("✅ **Great news!** The predicted fare is within your budget. Consider booking this flight!")
            else:
                st.markdown("⚠️ **Heads up!** The predicted fare exceeds your budget. Check out cheaper alternatives below.")
                
                st.subheader("Cheaper Alternatives")
                similar_flights = df[
                    (df['Source'] == label_encoders['Source'].transform([source_map[source_name]])[0]) &
                    (df['Destination'] == label_encoders['Destination'].transform([destination_map[destination_name]])[0]) &
                    (df['Total Fare (BDT)'] <= budget)
                ].sort_values('Total Fare (BDT)')
                
                if not similar_flights.empty:
                    st.markdown("Here are some flights within your budget:")
                    display_df = similar_flights.copy()
                    categorical_cols = ['Airline', 'Stopovers', 'Aircraft Type', 'Class', 'Booking Source', 'Seasonality']
                    for col in categorical_cols:
                        display_df[col] = label_encoders[col].inverse_transform(display_df[col].astype(int))
                    
                    display_columns = [
                        'Airline', 'Source Name', 'Destination Name', 'Duration (hrs)', 
                        'Stopovers', 'Aircraft Type', 'Class', 'Booking Source', 
                        'Seasonality', 'Days Before Departure', 'Total Fare (BDT)'
                    ]
                    st.dataframe(display_df[display_columns], use_container_width=True)
                else:
                    st.markdown("No cheaper flights found for this route within your budget.")

if __name__ == "__main__":
    main()

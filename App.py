import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
import uuid

# Function to load and preprocess data
def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv('Flight_Price_Dataset_of_Bangladesh.csv')
    
    # Create airport code to name mappings
    airport_mappings = df[['Source', 'Source Name', 'Destination', 'Destination Name']].drop_duplicates()
    source_map = dict(zip(airport_mappings['Source Name'], airport_mappings['Source']))
    destination_map = dict(zip(airport_mappings['Destination Name'], airport_mappings['Destination']))
    
    # Select relevant features
    features = ['Airline', 'Source', 'Destination', 'Duration (hrs)', 'Stopovers', 
                'Aircraft Type', 'Class', 'Booking Source', 'Seasonality', 'Days Before Departure']
    target = 'Total Fare (BDT)'
    
    # Handle categorical variables
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
    
    # Encode categorical inputs
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
    st.title("Flight Price Prediction App")
    st.write("Select flight details to predict the total fare in BDT.")

    # Load and preprocess data
    X, y, label_encoders, df, source_map, destination_map = load_and_preprocess_data()
    
    # Train the model
    model = train_model(X, y)
    
    # Create input form
    with st.form("flight_form"):
        st.subheader("Flight Details")
        
        # Dropdowns for categorical features
        airline = st.selectbox("Airline", options=label_encoders['Airline'].classes_)
        
        # Display full airport names in dropdowns
        source_name = st.selectbox("Source Airport", options=sorted(source_map.keys()))
        source_code = source_map[source_name]
        
        destination_name = st.selectbox("Destination Airport", options=sorted(destination_map.keys()))
        destination_code = destination_map[destination_name]
        
        stopovers = st.selectbox("Stopovers", options=label_encoders['Stopovers'].classes_)
        aircraft_type = st.selectbox("Aircraft Type", options=label_encoders['Aircraft Type'].classes_)
        class_type = st.selectbox("Class", options=label_encoders['Class'].classes_)
        booking_source = st.selectbox("Booking Source", options=label_encoders['Booking Source'].classes_)
        seasonality = st.selectbox("Seasonality", options=label_encoders['Seasonality'].classes_)
        
        # Numeric inputs
        duration = st.number_input("Duration (hours)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
        days_before = st.number_input("Days Before Departure", min_value=0, max_value=365, value=30, step=1)
        budget = st.number_input("Your Budget (BDT)", min_value=0.0, value=30000.0, step=1000.0)
        
        # Submit button
        submitted = st.form_submit_button("Predict Fare")
    
    if submitted:
        # Prepare input data with airport codes
        input_data = {
            'Airline': airline,
            'Source': source_code,
            'Destination': destination_code,
            'Duration (hrs)': duration,
            'Stopovers': stopovers,
            'Aircraft Type': aircraft_type,
            'Class': class_type,
            'Booking Source': booking_source,
            'Seasonality': seasonality,
            'Days Before Departure': days_before
        }
        
        # Predict fare
        predicted_fare = predict_fare(model, label_encoders, X.columns, input_data)
        
        if predicted_fare is not None:
            st.success(f"**Predicted Total Fare: {predicted_fare:.2f} BDT**")
            
            # Decision-making based on budget
            if predicted_fare <= budget:
                st.write("✅ The predicted fare is within your budget. Consider booking this flight!")
            else:
                st.write("⚠️ The predicted fare exceeds your budget. Explore cheaper alternatives.")
                
                # Suggest cheaper alternatives
                st.subheader("Cheaper Alternatives")
                # Filter flights with similar source/destination but lower fare
                similar_flights = df[
                    (df['Source'] == label_encoders['Source'].transform([source_code])[0]) &
                    (df['Destination'] == label_encoders['Destination'].transform([destination_code])[0]) &
                    (df['Total Fare (BDT)'] <= budget)
                ].sort_values('Total Fare (BDT)')
                
                if not similar_flights.empty:
                    st.write("Here are some flights within your budget:")
                    # Create a copy for display and inverse transform categorical columns
                    display_df = similar_flights.copy()
                    categorical_cols = ['Airline', 'Stopovers', 'Aircraft Type', 'Class', 'Booking Source', 'Seasonality']
                    for col in categorical_cols:
                        display_df[col] = label_encoders[col].inverse_transform(display_df[col].astype(int))
                    
                    # Select relevant columns for display
                    display_columns = [
                        'Airline', 'Source Name', 'Destination Name', 'Departure Date & Time', 'Arrival Date & Time',
                        'Duration (hrs)', 'Stopovers', 'Aircraft Type', 'Class', 'Booking Source', 
                        'Seasonality', 'Days Before Departure', 'Total Fare (BDT)'
                    ]
                    st.dataframe(display_df[display_columns])
                else:
                    st.write("No cheaper flights found for this route within your budget.")

if __name__ == "__main__":
    main()

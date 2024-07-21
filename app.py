
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import streamlit as st

# Function to load the CatBoostRegressor model (replace with your actual loading mechanism)
def load_model(filename="Two_Wheeler_Whprkm_Pred1.cbm"):
    """Loads the saved CatBoostRegressor model from a file (replace with your implementation)."""
    # Implement your model loading logic here (e.g., read data from file, create model object)
    # This might involve loading a serialized model or creating a new model from scratch
    # based on pre-saved information.
    model = None  # Placeholder for your loaded model
    print(f"Model loaded successfully from {filename}")  # Placeholder for success message
    return model


# 

# Function to load the CatBoostRegressor model (assuming "model.cbm" exists)
def load_model(filename="Two_Wheeler_Whprkm_Pred.cbm"):
    """Loads the saved CatBoostRegressor model from a file."""
    try:
        model = CatBoostRegressor()
        model.load_model(filename)
        print(f"Model loaded successfully from {filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        st.error("An error occurred while loading the model.")
        exit(1)  # Exit the script if model loading fails




# Function to preprocess user input data
def preprocess_data(data, selected_max_speed):
    """Preprocesses user input data according to your model's requirements."""

    # Convert data to a dictionary representing a DataFrame row
    df = {
        'b': data[0],
        'crr': data[1],
        'Peak Motor Power Kw': data[2],
        'drr': data[3],
        'Final Gear': data[4],
        'Independent_GVW': data[5],
    }

    # Create features based on user input
    features = ['b', 'crr', 'Peak Motor Power Kw', 'drr', 'Final Gear', 'Independent_GVW']
    df1 = {k: df[k] for k in features}

    # Handle missing values (replace with your strategy)
    # This might involve checking for missing values in the dictionary and imputing them
    # using a suitable method (e.g., mean, median, etc.)
    for key, value in df1.items():
        if value is None:
            # Implement your missing value handling logic here (e.g., df1[key] = some_imputation_method)
            pass  # Placeholder for missing value handling

    # Identify categorical features (replace with your logic for identifying categorical features)
    categorical_cols = ['Running Cycle']

    # One-hot encode categorical features (replace with your one-hot encoding implementation)
    # This might involve creating new keys in the dictionary for each category level
    # based on the selected 'Running Cycle' value.
    if 'Running Cycle' in data:
        running_cycle = data[6]
        df1['Running Cycle_IDC'] = 0 if running_cycle != 'IDC' else 1
        df1['Running Cycle_WMTC'] = 0 if running_cycle != 'WMTC' else 1
        df1['Running Cycle_PCMC'] = 0 if running_cycle != 'PCMC' else 1
    else:
        # Handle cases where 'Running Cycle' is not provided
        df1['Running Cycle_IDC'] = 0
        df1['Running Cycle_WMTC'] = 0
        df1['Running Cycle_PCMC'] = 0

    # Create dummy variables for selected Max_Speed (replace with your logic)
    # This might involve creating new keys in the dictionary for each category level
    # based on the selected 'selected_max_speed' value.
    max_speed_cols = [f'Max_Speed_{i}' for i in range(25, 51, 5)]
    df1[f'Max_Speed_{selected_max_speed}'] = 1
    for col in max_speed_cols:
        if col != f'Max_Speed_{selected_max_speed}':
            df1[col] = 0
    df = pd.DataFrame(df1, index=[0])  # Add an index (optional)

    # Return the preprocessed data
    return df

# def predict_energy_consumption(model, data, selected_max_speed):
#     """Makes predictions using the trained CatBoostRegressor model."""

#     preprocessed_data = preprocess_data(data, selected_max_speed)
#     X = preprocessed_data  # Use the entire preprocessed data for prediction
    
#     # Replace this with your actual prediction logic using the loaded model
#     y_pred = None  # Placeholder for prediction result
#     # Use your model to make a prediction based on the preprocessed data (X)
#     # and store the result in y_pred.

#     return y_pred




def predict_energy_consumption(model, data, selected_max_speed):
    """Makes predictions using the trained CatBoostRegressor model."""

    preprocessed_data = preprocess_data(data, selected_max_speed)
    cols = ['b', 'crr', 'Peak Motor Power Kw', 'drr', 'Final Gear',
       'Independent_GVW', 'Running Cycle_IDC', 'Running Cycle_WMTC',
       'Max_Speed_25', 'Max_Speed_30', 'Max_Speed_35', 'Max_Speed_40',
       'Max_Speed_45']
    
    # Select columns using intersection
    available_cols = set(preprocessed_data.columns) & set(cols)  # Intersection of column sets
    X = preprocessed_data[list(available_cols)]  # Convert back to list for indexing

    
    # X = preprocessed_data[cols]  # Use the entire preprocessed data for prediction
    X_array = X.to_numpy()
    y_pred = round(model.predict(X_array)[0],1)  # Make a single prediction

    return y_pred



# Load the saved model (assuming "model.cbm" exists)
try:
    model = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    st.error("An error occurred while loading the model.")
    exit(1)  # Exit the script if model loading fails

st.title("Energy Consumption Prediction App")

# User input for non-Max_Speed features
# b = float(input("Aerodynamic Coeficient (b): "))  # Assuming user input is a number
# crr = float(input("crr [N/kg]: "))
# peak_motor_power = float(input("Peak Motor Power Kw: "))
# drr = float(input("DRR [m]: "))
# final_gear = float(input("Final Gear: "))
# independent_gvw = float(input("Independent_GVW: "))
# Battery_Energy = float(input("Battery Energy: "))
# DOD = float(input("DOD: "))




# Assuming user input for Running Cycle and other features is handled elsewhere (replace with your logic)
b = st.number_input("Aerodynamic Coeficient (b): ", min_value=0.0)  # Add min/max values if applicable

crr = 0.18
peak_motor_power =4.2
drr = 0.221
final_gear = 9.82
independent_gvw = 240
Battery_Energy = 2.8
DOD = 90






running_cycle = "IDC"  # Placeholder for user input (replace with actual logic)
auxiliary_power = 0.0  # Placeholder value (replace with actual logic)
gear_eff = 0.93  # Placeholder value (replace with actual logic)




# Prepare data for prediction (replace with your actual logic)
user_data = [b, crr, peak_motor_power, drr, final_gear, independent_gvw, running_cycle, auxiliary_power, gear_eff]
selected_max_speed = "35"  # Placeholder for user selection (replace with actual logic)

# Make prediction (replace with your actual prediction logic)
prediction = predict_energy_consumption(model, user_data, selected_max_speed)

# Display prediction result (replace with your implementation)
# print(f"Predicted Energy Consumption: {prediction}")

st.write(f"Predicted Energy Consumption: {prediction}")

# Range = Battery_Energy*1000*DOD/prediction/100
# # Display prediction result (replace with your implementation)
# print(f"Predicted Range: {Range}")

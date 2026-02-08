import pickle
import pandas as pd

# Load model
with open("real_estate_xgb_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
encoder = artifact["encoder"]
features = artifact["features"]

# ============================================
# SINGLE PROPERTY PREDICTION
# ============================================
# Create input data
property_data = pd.DataFrame({
    'Town': ['Hartford'],
    'Residential Type': ['Single Family'],
    'Assessed Value': [300000],
    'Sales Ratio': [0.95],
    'Years_Since_List': [2]
})

# Encode and predict
property_data[['Town', 'Residential Type']] = encoder.transform(property_data[['Town', 'Residential Type']])
prediction = model.predict(property_data[features])

print(f"🏠 Predicted Price: ${prediction[0]:,.2f}")

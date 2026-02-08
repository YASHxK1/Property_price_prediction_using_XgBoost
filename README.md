# 🏠 Property Price Prediction using XGBoost

A machine learning project that predicts real estate sale prices using XGBoost regression, trained on Connecticut real estate sales data from 2001-2023.

## 📊 Project Overview

This project implements a property price prediction model using the XGBoost algorithm. The model analyzes various property features including location, residential type, assessed value, and sales ratio to predict accurate sale prices.

### Key Features

- **Dataset**: 1.14M+ real estate transactions (2001-2023)
- **Algorithm**: XGBoost Regressor with optimized hyperparameters
- **Performance**: R² Score of 0.6157
- **Features Used**: Town, Residential Type, Assessed Value, Sales Ratio, Years Since List

## 📁 Project Structure

```
Property_price_prediction_using_XgBoost/
├── model_XGB_training.ipynb          # Main training notebook
├── use_model_XGB.py                  # Inference script
├── real_estate_xgb_model.pkl         # Trained model (2.3 MB)
└── Real_Estate_Sales_2001_to_2023.csv# Dataset
```
## dataset link
'''
https://www.kaggle.com/datasets/yasmeenfahme/real-estate-sales
'''
## or
''' 
https://catalog.data.gov/dataset/real-estate-sales-2001-2018
'''

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy xgboost scikit-learn matplotlib pickle
```

### Dataset

The project uses the **Real Estate Sales 2001-2023** dataset from Connecticut, containing:
- 1,141,722 initial records
- 14 columns including property details, sale information, and location data
- After cleaning: 738,541 valid records

## 🔧 Model Training

### Data Preprocessing

1. **Date Parsing**: Convert date strings to datetime objects
2. **Missing Value Handling**: Drop rows with missing critical values
3. **Feature Engineering**: Create `Years_Since_List` feature
4. **Encoding**: OrdinalEncoder for categorical features (Town, Residential Type)
5. **Train-Test Split**: 80-20 split (590,832 training / 147,709 test samples)

### Model Configuration

```python
XGBRegressor(
    tree_method='hist',
    max_depth=3,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
```

### Training Process

Run the Jupyter notebook:

```bash
jupyter notebook model_XGB_training.ipynb
```

The training process includes:
- Data loading and cleaning
- Feature engineering
- Model training with validation
- Performance evaluation
- Model serialization

## 📈 Model Performance

| Metric       | Value       |
| ------------ | ----------- |
| **R² Score** | 0.6157      |
| **MAE**      | $35,357.04  |
| **RMSE**     | $476,968.96 |

### Feature Importance

| Feature          | Importance |
| ---------------- | ---------- |
| Town             | 37.56%     |
| Assessed Value   | 33.33%     |
| Sales Ratio      | 16.73%     |
| Residential Type | 12.09%     |
| Years Since List | 0.30%      |

## 💡 Usage

### Making Predictions

```python
import pickle
import pandas as pd

# Load the trained model
with open("real_estate_xgb_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
encoder = artifact["encoder"]
features = artifact["features"]

# Prepare input data
property_data = pd.DataFrame({
    'Town': ['Hartford'],
    'Residential Type': ['Single Family'],
    'Assessed Value': [300000],
    'Sales Ratio': [0.95],
    'Years_Since_List': [2]
})

# Encode categorical features
property_data[['Town', 'Residential Type']] = encoder.transform(
    property_data[['Town', 'Residential Type']]
)

# Make prediction
prediction = model.predict(property_data[features])
print(f"🏠 Predicted Price: ${prediction[0]:,.2f}")
```

Or simply run the provided script:

```bash
python use_model_XGB.py
```

## 📊 Data Features

### Input Features

1. **Town**: Property location (categorical)
2. **Residential Type**: Type of residence (categorical)
3. **Assessed Value**: Official property assessment value
4. **Sales Ratio**: Ratio of assessed value to sale price
5. **Years_Since_List**: Years between listing and sale

### Target Variable

- **Sale Amount**: Actual property sale price

## 🎯 Model Insights

- **Town** is the most important feature (37.56%), indicating location is crucial for price prediction
- **Assessed Value** closely follows (33.33%), showing official assessments are strong predictors
- The model achieves reasonable accuracy with an R² of 0.6157
- RMSE of ~$477K suggests the model works better for lower-priced properties

## 🔄 Model Artifacts

The saved model file (`real_estate_xgb_model.pkl`) contains:

```python
{
    "model": XGBRegressor,           # Trained XGBoost model
    "encoder": OrdinalEncoder,       # Fitted encoder for categorical features
    "features": list,                # Feature names in correct order
    "metrics": {
        "r2": float,                 # R² score
        "mae": float,                # Mean Absolute Error
        "rmse": float                # Root Mean Squared Error
    }
}
```

## 📝 Training Details

### Hyperparameters

- **Tree Method**: Histogram-based algorithm
- **Max Depth**: 3 (prevents overfitting)
- **Learning Rate**: 0.1
- **Number of Estimators**: 200 trees
- **Subsample**: 0.8 (80% of data per tree)
- **Column Sample**: 0.8 (80% of features per tree)

### Validation

- Early stopping monitoring with test set evaluation
- 200 training rounds with RMSE tracking
- Final validation RMSE: $476,968.96

## 🛠️ Future Improvements

- [ ] Feature engineering: Add more derived features (e.g., price per square foot)
- [ ] Hyperparameter tuning using GridSearchCV or Optuna
- [ ] Ensemble methods: Combine with other algorithms
- [ ] Add property size/square footage data if available
- [ ] Implement cross-validation for more robust evaluation
- [ ] Deploy as a web API using Flask/FastAPI

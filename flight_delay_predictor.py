# %%
import glob
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# %%
# Configuration - set your data directory here
data_dir = "data"  # Use "data" for full dataset or "data/sample" for sample data

# Load the processed weather and delay data
csv_files = glob.glob(f"{data_dir}/weather_delay.csv/*.part")

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
df.head()

# %%
# Data preprocessing and feature engineering
print("Data types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# %%
# Feature selection - focus on weather features for predicting arrival delay
weather_features = [
    "dep_temp",
    "dep_dwpt",
    "dep_rhum",
    "dep_prcp",
    "dep_snow",
    "dep_wdir",
    "dep_wspd",
    "dep_wpgt",
    "dep_pres",
    "dep_coco",
    "arr_temp",
    "arr_dwpt",
    "arr_rhum",
    "arr_prcp",
    "arr_snow",
    "arr_wdir",
    "arr_wspd",
    "arr_wpgt",
    "arr_pres",
    "arr_coco",
]

categorical_features = ["Origin", "Dest"]
target = "ArrDelayMinutes"

# Check which weather features exist in the dataset
existing_weather_features = [col for col in weather_features if col in df.columns]
print(f"Available weather features: {existing_weather_features}")

# %%
# Encode categorical features
le_origin = LabelEncoder()
le_dest = LabelEncoder()

df_processed = df.copy()
df_processed["Origin_encoded"] = le_origin.fit_transform(df_processed["Origin"])
df_processed["Dest_encoded"] = le_dest.fit_transform(df_processed["Dest"])

# Create feature matrix
feature_columns = existing_weather_features + [
    "Origin_encoded",
    "Dest_encoded",
]
X = df_processed[feature_columns].copy()
y = df_processed[target].copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# %%
# Handle missing values
# First, identify columns that are completely missing and remove them
non_null_cols = X.columns[X.notna().any()].tolist()
X_filtered = X[non_null_cols].copy()

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_filtered),
    columns=X_filtered.columns,
    index=X_filtered.index,
)

# Remove rows where target is missing
mask = ~y.isnull()
X_clean = X_imputed[mask]
y_clean = y[mask]

print(f"Clean dataset shape: X={X_clean.shape}, y={y_clean.shape}")
print("Target statistics:")
print(y_clean.describe())

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

# Update feature columns to match the filtered dataset
feature_columns = X_clean.columns.tolist()

# Ensure we have DataFrame objects
X_train = pd.DataFrame(X_train, columns=X_clean.columns)
X_test = pd.DataFrame(X_test, columns=X_clean.columns)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# %%
# Scale features for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Train multiple models
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Linear Regression": LinearRegression(),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "predictions": y_pred,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
    }

    print(f"{name} Results:")
    print(f"  MAE: {mae:.2f} minutes")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  R²: {r2:.4f}")

# %%
# Feature importance analysis (Random Forest)
rf_model = results["Random Forest"]["model"]
feature_importance = pd.DataFrame(
    {"feature": X_clean.columns, "importance": rf_model.feature_importances_}
).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(15), x="importance", y="feature")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

print("Top 10 most important features:")
print(feature_importance.head(10))

# %%
# Prediction accuracy visualization
best_model_name = max(results.keys(), key=lambda k: results[k]["R²"])
best_predictions = results[best_model_name]["predictions"]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
y_test_array = np.array(y_test)
plt.scatter(y_test_array, best_predictions, alpha=0.5)
min_val, max_val = float(y_test.min()), float(y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
plt.xlabel("Actual Delay (minutes)")
plt.ylabel("Predicted Delay (minutes)")
plt.title(f"{best_model_name}: Actual vs Predicted")

plt.subplot(1, 2, 2)
residuals = y_test_array - best_predictions
plt.scatter(best_predictions, residuals, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted Delay (minutes)")
plt.ylabel("Residuals (minutes)")
plt.title(f"{best_model_name}: Residual Plot")

plt.tight_layout()
plt.show()

# %%
# Delay distribution analysis
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(y_clean, bins=50, alpha=0.7, edgecolor="black")
plt.xlabel("Arrival Delay (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Arrival Delays")

plt.subplot(1, 3, 2)
plt.hist(y_clean[y_clean <= 60], bins=30, alpha=0.7, edgecolor="black")
plt.xlabel("Arrival Delay (minutes)")
plt.ylabel("Frequency")
plt.title("Delays ≤ 60 minutes")

plt.subplot(1, 3, 3)
delay_categories = pd.cut(
    y_clean,
    bins=[-np.inf, -15, 0, 15, 60, np.inf],
    labels=[
        "Early (>15min)",
        "Early (≤15min)",
        "On Time to +15min",
        "Delayed (15-60min)",
        "Severely Delayed (>60min)",
    ],
)
delay_categories.value_counts().plot(kind="bar", rot=45)
plt.title("Delay Categories")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# %%
# Weather impact analysis
weather_corr = (
    df_processed[existing_weather_features + [target]].corr()[target].drop(target)
)
weather_corr_sorted = weather_corr.abs().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
weather_corr_sorted.head(10).plot(kind="bar")
plt.title("Top 10 Weather Features Correlation with Arrival Delay")
plt.ylabel("Absolute Correlation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Weather features most correlated with arrival delays:")
print(weather_corr_sorted.head(10))

# %%
# Model comparison summary
print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)

comparison_df = pd.DataFrame(
    {
        name: {
            "MAE (minutes)": f"{results[name]['MAE']:.2f}",
            "RMSE (minutes)": f"{results[name]['RMSE']:.2f}",
            "R² Score": f"{results[name]['R²']:.4f}",
        }
        for name in results.keys()
    }
).T

print(comparison_df)

print(f"\nBest performing model: {best_model_name}")
print(f"Best R² Score: {results[best_model_name]['R²']:.4f}")

# %%
# Save the best model and preprocessing objects

# Create model directory
model_dir = f"{data_dir}/model"
os.makedirs(model_dir, exist_ok=True)

# Save model, scaler, and encoders
joblib.dump(
    results[best_model_name]["model"], f"{model_dir}/best_flight_delay_model.pkl"
)
if best_model_name == "Linear Regression":
    joblib.dump(scaler, f"{model_dir}/feature_scaler.pkl")
joblib.dump(le_origin, f"{model_dir}/origin_encoder.pkl")
joblib.dump(le_dest, f"{model_dir}/dest_encoder.pkl")
joblib.dump(imputer, f"{model_dir}/feature_imputer.pkl")

print(f"\nModel artifacts saved to {model_dir}/:")
print("- best_flight_delay_model.pkl")
if best_model_name == "Linear Regression":
    print("- feature_scaler.pkl")
print("- origin_encoder.pkl")
print("- dest_encoder.pkl")
print("- feature_imputer.pkl")


# %%
# Example prediction function
def predict_flight_delay(origin, dest, weather_data, model_data_dir=None):
    """
    Predict flight arrival delay based on origin, destination, and weather data.

    Parameters:
    - origin: IATA airport code (str)
    - dest: IATA airport code (str)
    - weather_data: Dictionary with weather features
    - model_data_dir: Data directory containing model files (defaults to data_dir used in training)

    Returns:
    - Predicted arrival delay in minutes
    """
    if model_data_dir is None:
        model_data_dir = data_dir
    # Load saved objects
    model_dir = f"{model_data_dir}/model"
    model = joblib.load(f"{model_dir}/best_flight_delay_model.pkl")
    imputer = joblib.load(f"{model_dir}/feature_imputer.pkl")
    le_origin = joblib.load(f"{model_dir}/origin_encoder.pkl")
    le_dest = joblib.load(f"{model_dir}/dest_encoder.pkl")

    # Prepare input data
    input_data = weather_data.copy()

    # Handle unknown airports
    try:
        input_data["Origin_encoded"] = le_origin.transform([origin])[0]
    except ValueError:
        input_data["Origin_encoded"] = -1  # Unknown airport

    try:
        input_data["Dest_encoded"] = le_dest.transform([dest])[0]
    except ValueError:
        input_data["Dest_encoded"] = -1  # Unknown airport

    # No departure delay needed - using only weather and route

    # Create feature vector
    input_df = pd.DataFrame([input_data])[feature_columns]
    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

    # Scale if needed
    if best_model_name == "Linear Regression":
        scaler = joblib.load(f"{model_dir}/feature_scaler.pkl")
        input_imputed = scaler.transform(input_imputed)

    # Make prediction
    prediction = model.predict(input_imputed)[0]
    return prediction


print("\nExample usage:")
print("predicted_delay = predict_flight_delay('DEN', 'DAY', weather_data_dict)")

# %%
# Quick test of the prediction function
print("\n" + "=" * 50)
print("TESTING PREDICTION FUNCTION")
print("=" * 50)

# Sample weather data for testing
test_weather_data = {
    "dep_temp": 20.5,
    "dep_dwpt": 15.2,
    "dep_rhum": 65.0,
    "dep_prcp": 0.0,
    "dep_wdir": 270.0,
    "dep_wspd": 15.0,
    "dep_pres": 1013.2,
    "dep_coco": 1.0,
    "arr_temp": 18.3,
    "arr_dwpt": 12.1,
    "arr_rhum": 58.0,
    "arr_prcp": 0.0,
    "arr_wdir": 180.0,
    "arr_wspd": 8.0,
    "arr_pres": 1015.8,
    "arr_coco": 2.0,
}

# Test scenarios
test_scenarios = [
    {
        "name": "Clear Weather",
        "weather": test_weather_data,
    },
    {
        "name": "Stormy Weather",
        "weather": {
            **test_weather_data,
            "dep_prcp": 10.0,  # Heavy rain
            "dep_wspd": 35.0,  # Strong winds
            "dep_coco": 4.0,  # Storm conditions
            "arr_prcp": 5.0,  # Rain at destination
            "arr_wspd": 25.0,  # Windy at destination
        },
    },
]

for scenario in test_scenarios:
    try:
        prediction = predict_flight_delay("DEN", "DAY", scenario["weather"])
        print(f"\n{scenario['name']}:")
        print(f"  Predicted delay: {prediction:.1f} minutes")

        if prediction < 0:
            print("  → Early arrival expected")
        elif prediction < 15:
            print("  → On time or minor delay")
        elif prediction < 60:
            print("  → Moderate delay expected")
        else:
            print("  → Significant delay expected")

    except Exception as e:
        print(f"\n{scenario['name']}: ERROR - {e}")

print(f"\n{'=' * 50}")
print("Prediction function test completed!")

# %%
import glob
import os
import sys
import warnings
from datetime import datetime

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Check if running in interactive mode
INTERACTIVE_MODE = (
    hasattr(sys, "ps1") or "ipykernel" in sys.modules or "jupyter" in sys.modules
)

if not INTERACTIVE_MODE:
    matplotlib.use("Agg")  # Use non-interactive backend


def show_or_save_plot(filename=None):
    """Show plot in interactive mode, save to file in batch mode"""
    if INTERACTIVE_MODE:
        plt.show()
    else:
        if filename:
            os.makedirs(f"{data_dir}/plots", exist_ok=True)
            plt.savefig(f"{data_dir}/plots/{filename}", dpi=300, bbox_inches="tight")
            print(f"Plot saved to {data_dir}/plots/{filename}")
        plt.close()


# %%
# Check device availability for M-series Mac
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


device = get_device()
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

# %%
# Configuration - set your data directory here
data_dir = "data"  # Use "data" for full dataset or "data/sample" for sample data

# Dataset sampling for faster training/testing (set to 1.0 for full dataset)
DATASET_SAMPLE_FRACTION = 0.10  # Use 10% of data by default

# Load the processed weather and delay data
csv_files = glob.glob(f"{data_dir}/weather_delay.csv/*.part")

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Sample dataset if requested
if DATASET_SAMPLE_FRACTION < 1.0:
    original_size = len(df)
    df = df.sample(frac=DATASET_SAMPLE_FRACTION, random_state=42).reset_index(drop=True)
    print(
        f"Sampled {DATASET_SAMPLE_FRACTION * 100:.1f}% of dataset: {len(df):,} rows (from {original_size:,})"
    )

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
# Scale features for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)

print(f"Training tensors: X={X_train_tensor.shape}, y={y_train_tensor.shape}")
print(f"Test tensors: X={X_test_tensor.shape}, y={y_test_tensor.shape}")


# %%
# Define Neural Network Architecture
class FlightDelayPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(FlightDelayPredictor, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers with stronger regularization
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Better initialization for regression
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


# Create model
input_dim = X_train_tensor.shape[1]
model = FlightDelayPredictor(input_dim).to(device)

print("Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")


# %%
# Training Configuration
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# Training hyperparameters - reduced overfitting
batch_size = 256
learning_rate = 0.001
num_epochs = 300

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Loss function and optimizer with better regularization
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
)

# Early stopping with more patience
early_stopping = EarlyStopping(patience=25, min_delta=1.0)

print("Training configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Max epochs: {num_epochs}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")


# %%
# Training Loop
def train_model():
    train_losses = []
    val_losses = []

    model.train()
    best_val_loss = float("inf")
    best_model_state = None

    # Progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0

        # Progress bar for training batches
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1} Train", leave=False, unit="batch"
        )
        for batch_X, batch_y in train_pbar:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            # Update progress bar
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            # Progress bar for validation batches
            val_pbar = tqdm(
                test_loader, desc=f"Epoch {epoch + 1} Val", leave=False, unit="batch"
            )
            for batch_X, batch_y in val_pbar:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                # Update progress bar
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()

        # Update epoch progress bar
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{avg_train_loss:.4f}",
                "val_loss": f"{avg_val_loss:.4f}",
                "lr": f"{current_lr:.6f}",
            }
        )

        # Early stopping
        if early_stopping(avg_val_loss):
            epoch_pbar.set_description(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


print("Starting training...")
start_time = datetime.now()
train_losses, val_losses = train_model()
end_time = datetime.now()
training_time = end_time - start_time

print(f"\nTraining completed in {training_time}")
print(f"Final train loss: {train_losses[-1]:.4f}")
print(f"Final validation loss: {val_losses[-1]:.4f}")

# %%
# Training Progress Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss", alpha=0.7)
plt.plot(val_losses, label="Validation Loss", alpha=0.7)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Progress")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_losses[10:], label="Training Loss", alpha=0.7)
plt.plot(val_losses[10:], label="Validation Loss", alpha=0.7)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Progress (After Epoch 10)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
show_or_save_plot("training_progress.png")

# %%
# Model Evaluation
model.eval()

with torch.no_grad():
    # Get predictions
    train_predictions = model(X_train_tensor).cpu().numpy().flatten()
    test_predictions = model(X_test_tensor).cpu().numpy().flatten()

# Calculate metrics
train_mae = mean_absolute_error(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, train_predictions)

test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_predictions)

print("=" * 60)
print("PYTORCH NEURAL NETWORK RESULTS")
print("=" * 60)
print("Training Results:")
print(f"  MAE: {train_mae:.2f} minutes")
print(f"  RMSE: {train_rmse:.2f} minutes")
print(f"  R²: {train_r2:.4f}")

print("\nTest Results:")
print(f"  MAE: {test_mae:.2f} minutes")
print(f"  RMSE: {test_rmse:.2f} minutes")
print(f"  R²: {test_r2:.4f}")

# %%
# Prediction Accuracy Visualization
plt.figure(figsize=(15, 5))

# Training set predictions
plt.subplot(1, 3, 1)
plt.scatter(y_train, train_predictions, alpha=0.5, s=1)
min_val, max_val = float(y_train.min()), float(y_train.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
plt.xlabel("Actual Delay (minutes)")
plt.ylabel("Predicted Delay (minutes)")
plt.title(f"Training Set: R² = {train_r2:.4f}")
plt.grid(True, alpha=0.3)

# Test set predictions
plt.subplot(1, 3, 2)
plt.scatter(y_test, test_predictions, alpha=0.5, s=1)
min_val, max_val = float(y_test.min()), float(y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
plt.xlabel("Actual Delay (minutes)")
plt.ylabel("Predicted Delay (minutes)")
plt.title(f"Test Set: R² = {test_r2:.4f}")
plt.grid(True, alpha=0.3)

# Residuals
plt.subplot(1, 3, 3)
residuals = y_test.values - test_predictions
plt.scatter(test_predictions, residuals, alpha=0.5, s=1)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted Delay (minutes)")
plt.ylabel("Residuals (minutes)")
plt.title("Residual Plot")
plt.grid(True, alpha=0.3)

plt.tight_layout()
show_or_save_plot("prediction_accuracy.png")


# %%
# Feature Importance Analysis using Permutation
def calculate_feature_importance(model, X_tensor, y_tensor, feature_names):
    model.eval()

    # Baseline performance
    with torch.no_grad():
        baseline_pred = model(X_tensor)
        baseline_mae = mean_absolute_error(
            y_tensor.cpu().numpy(), baseline_pred.cpu().numpy()
        )

    importances = []

    # Progress bar for feature importance calculation
    feature_pbar = tqdm(
        enumerate(feature_names),
        total=len(feature_names),
        desc="Calculating feature importance",
        unit="feature",
    )

    for i, feature_name in feature_pbar:
        # Create permuted version
        X_permuted = X_tensor.clone()
        perm_idx = torch.randperm(X_permuted.shape[0])
        X_permuted[:, i] = X_permuted[perm_idx, i]

        # Calculate performance with permuted feature
        with torch.no_grad():
            permuted_pred = model(X_permuted)
            permuted_mae = mean_absolute_error(
                y_tensor.cpu().numpy(), permuted_pred.cpu().numpy()
            )

        # Feature importance is the increase in error
        importance = permuted_mae - baseline_mae
        importances.append(importance)

        # Update progress bar
        feature_pbar.set_postfix({"current_feature": feature_name[:10]})

    return np.array(importances)


print("Calculating feature importance (this may take a moment)...")
feature_importance = calculate_feature_importance(
    model, X_test_tensor, y_test_tensor, feature_columns
)

feature_importance_df = pd.DataFrame(
    {"feature": feature_columns, "importance": feature_importance}
).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 8))
top_features = feature_importance_df.head(15)
sns.barplot(data=top_features, x="importance", y="feature")
plt.title("Top 15 Feature Importances (Permutation Method)")
plt.xlabel("Importance (MAE Increase)")
plt.tight_layout()
show_or_save_plot("feature_importance.png")

print("Top 10 most important features:")
print(feature_importance_df.head(10))

# %%
# Delay Distribution Analysis
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
show_or_save_plot("delay_distribution.png")

# %%
# Weather Impact Analysis
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
show_or_save_plot("weather_correlation.png")

print("Weather features most correlated with arrival delays:")
print(weather_corr_sorted.head(10))

# %%
# Save the model and preprocessing objects
model_dir = f"{data_dir}/model"
os.makedirs(model_dir, exist_ok=True)

# Save PyTorch model
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_architecture": {
            "input_dim": input_dim,
            "hidden_dims": [128, 64, 32],
            "dropout_rate": 0.3,
        },
        "feature_columns": feature_columns,
        "training_stats": {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mae": train_mae,
            "test_mae": test_mae,
        },
    },
    f"{model_dir}/pytorch_flight_delay_model.pth",
)

# Save preprocessing objects
joblib.dump(scaler, f"{model_dir}/feature_scaler.pkl")
joblib.dump(le_origin, f"{model_dir}/origin_encoder.pkl")
joblib.dump(le_dest, f"{model_dir}/dest_encoder.pkl")
joblib.dump(imputer, f"{model_dir}/feature_imputer.pkl")

print(f"\nModel artifacts saved to {model_dir}/:")
print("- pytorch_flight_delay_model.pth")
print("- feature_scaler.pkl")
print("- origin_encoder.pkl")
print("- dest_encoder.pkl")
print("- feature_imputer.pkl")


# %%
# Enhanced Prediction Function for PyTorch Model
def predict_flight_delay(origin, dest, weather_data, model_data_dir=None):
    """
    Predict flight arrival delay using PyTorch neural network.

    Parameters:
    - origin: IATA airport code (str)
    - dest: IATA airport code (str)
    - weather_data: Dictionary with weather features
    - model_data_dir: Data directory containing model files

    Returns:
    - Predicted arrival delay in minutes
    """
    if model_data_dir is None:
        model_data_dir = data_dir

    model_dir = f"{model_data_dir}/model"

    # Load model
    checkpoint = torch.load(
        f"{model_dir}/pytorch_flight_delay_model.pth", map_location=device
    )

    # Recreate model architecture
    model_arch = checkpoint["model_architecture"]
    loaded_model = FlightDelayPredictor(
        model_arch["input_dim"], model_arch["hidden_dims"], model_arch["dropout_rate"]
    ).to(device)

    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()

    # Load preprocessing objects
    scaler = joblib.load(f"{model_dir}/feature_scaler.pkl")
    le_origin = joblib.load(f"{model_dir}/origin_encoder.pkl")
    le_dest = joblib.load(f"{model_dir}/dest_encoder.pkl")
    imputer = joblib.load(f"{model_dir}/feature_imputer.pkl")

    # Prepare input data
    input_data = weather_data.copy()

    # Handle unknown airports
    try:
        input_data["Origin_encoded"] = le_origin.transform([origin])[0]
    except ValueError:
        input_data["Origin_encoded"] = -1

    try:
        input_data["Dest_encoded"] = le_dest.transform([dest])[0]
    except ValueError:
        input_data["Dest_encoded"] = -1

    # Create feature vector
    feature_cols = checkpoint["feature_columns"]
    input_df = pd.DataFrame([input_data])[feature_cols]
    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

    # Scale features
    input_scaled = scaler.transform(input_imputed)

    # Convert to tensor and predict
    input_tensor = torch.FloatTensor(input_scaled).to(device)

    with torch.no_grad():
        prediction = loaded_model(input_tensor).cpu().numpy()[0][0]

    return prediction


print("\nEnhanced PyTorch prediction function ready!")
print("Usage: predicted_delay = predict_flight_delay('DEN', 'DAY', weather_data_dict)")

# %%
# Test the PyTorch prediction function
print("\n" + "=" * 60)
print("TESTING PYTORCH PREDICTION FUNCTION")
print("=" * 60)

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
            "dep_prcp": 10.0,
            "dep_wspd": 35.0,
            "dep_coco": 4.0,
            "arr_prcp": 5.0,
            "arr_wspd": 25.0,
        },
    },
    {
        "name": "Cold Weather",
        "weather": {
            **test_weather_data,
            "dep_temp": -10.0,
            "dep_dwpt": -15.0,
            "arr_temp": -8.0,
            "arr_dwpt": -12.0,
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

print(f"\n{'=' * 60}")
print("PyTorch Neural Network Performance Summary:")
print(f"  Test R² Score: {test_r2:.4f}")
print(f"  Test MAE: {test_mae:.2f} minutes")
print(f"  Test RMSE: {test_rmse:.2f} minutes")
print(f"  Training Time: {training_time}")
print(f"  Device Used: {device}")
print("=" * 60)

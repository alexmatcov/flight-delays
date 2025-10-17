# %%
import io
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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from storage import config, Storage

warnings.filterwarnings("ignore")

INTERACTIVE_MODE = (
    hasattr(sys, "ps1") or "ipykernel" in sys.modules or "jupyter" in sys.modules
)
if not INTERACTIVE_MODE:
    matplotlib.use("Agg")


def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


storage = Storage()


def show_or_save_plot(filename=None):
    if INTERACTIVE_MODE:
        plt.show()
    else:
        if filename:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches="tight")
            buf.seek(0)
            
            s3_key = f"{config.results_path}/plots/{filename}"
            storage.s3.upload_fileobj(buf, config.bucket_name, s3_key)
            print(f"Saved plot to s3://{config.bucket_name}/{s3_key}")
        plt.close()


device = get_device()

# %%
DATASET_SAMPLE_FRACTION = 0.10

data_path = config.get_s3_path(f"{config.processed_path}/weather_delay_merged.parquet")
df = pd.read_parquet(data_path, storage_options=config.s3fs_storage_options)

if DATASET_SAMPLE_FRACTION < 1.0:
    original_size = len(df)
    df = df.sample(frac=DATASET_SAMPLE_FRACTION, random_state=42).reset_index(drop=True)
    print(
        f"Sampled {DATASET_SAMPLE_FRACTION * 100:.1f}% of dataset: {len(df):,} rows (from {original_size:,})"
    )

print(f"Dataset shape: {df.shape}")

# %%
weather_features = [
    "dep_temp",
    "dep_dwpt",
    "dep_rhum",
    "dep_prcp",
    "dep_wdir",
    "dep_wspd",
    "dep_pres",
    "dep_coco",
    "arr_temp",
    "arr_dwpt",
    "arr_rhum",
    "arr_prcp",
    "arr_wdir",
    "arr_wspd",
    "arr_pres",
    "arr_coco",
]

df_processed = df.copy()

# Feature engineering
df_processed["temp_diff"] = abs(df_processed["dep_temp"] - df_processed["arr_temp"])
df_processed["total_prcp"] = df_processed["dep_prcp"].fillna(0) + df_processed[
    "arr_prcp"
].fillna(0)
df_processed["dep_wind_stress"] = df_processed["dep_wspd"] * (
    df_processed["dep_coco"] / 10.0
)
df_processed["arr_wind_stress"] = df_processed["arr_wspd"] * (
    df_processed["arr_coco"] / 10.0
)
df_processed["pressure_diff"] = abs(df_processed["dep_pres"] - df_processed["arr_pres"])
df_processed["dep_bad_weather"] = (
    (df_processed["dep_prcp"].fillna(0) > 5)
    | (df_processed["dep_wspd"] > 30)
    | (df_processed["dep_coco"] >= 6)
).astype(int)
df_processed["arr_bad_weather"] = (
    (df_processed["arr_prcp"].fillna(0) > 5)
    | (df_processed["arr_wspd"] > 30)
    | (df_processed["arr_coco"] >= 6)
).astype(int)

engineered_features = [
    "temp_diff",
    "total_prcp",
    "dep_wind_stress",
    "arr_wind_stress",
    "pressure_diff",
    "dep_bad_weather",
    "arr_bad_weather",
]

le_origin = LabelEncoder()
le_dest = LabelEncoder()
df_processed["Origin_encoded"] = le_origin.fit_transform(df_processed["Origin"])
df_processed["Dest_encoded"] = le_dest.fit_transform(df_processed["Dest"])

existing_weather_features = [
    col for col in weather_features if col in df_processed.columns
]
existing_engineered_features = [
    col for col in engineered_features if col in df_processed.columns
]

feature_columns = (
    existing_weather_features
    + existing_engineered_features
    + ["Origin_encoded", "Dest_encoded"]
)
X = df_processed[feature_columns].copy()
y = df_processed["ArrDelayMinutes"].copy()

print(f"Features: {len(feature_columns)}, Samples: {len(X)}")


# %%
def display_correlation_matrix(
    X_data, y_data, feature_names=None, title="Feature Correlation Matrix"
):
    """Display correlation matrix for features and target variable."""
    if feature_names is None:
        feature_names = X_data.columns.tolist()

    corr_data = X_data.copy()
    corr_data["target"] = y_data

    corr_matrix = corr_data.corr()

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        annot=False,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    show_or_save_plot("correlation_matrix.png")

    target_corr = corr_matrix["target"].drop("target").sort_values(ascending=False)
    print(
        f"\nHighest positive correlation with target: {target_corr.idxmax()} ({target_corr.max():.3f})"
    )
    print(
        f"Highest negative correlation with target: {target_corr.idxmin()} ({target_corr.idxmin()})"
    )

    return corr_matrix


# Display correlation matrix for all features
print("\n" + "=" * 60)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 60)

correlation_matrix = display_correlation_matrix(X, y, feature_columns)

# %%
non_null_cols = X.columns[X.notna().any()].tolist()
X_filtered = X[non_null_cols].copy()

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_filtered),
    columns=X_filtered.columns,
    index=X_filtered.index,
)

mask = ~y.isnull()
X_clean = X_imputed[mask]
y_clean = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

# Ensure proper pandas types
X_train = pd.DataFrame(X_train, columns=X_clean.columns)
X_test = pd.DataFrame(X_test, columns=X_clean.columns)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)

feature_columns = X_clean.columns.tolist()


class FlightDelayPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(FlightDelayPredictor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


class EarlyStopping:
    def __init__(self, patience=25, min_delta=1.0):
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


input_dim = X_train_tensor.shape[1]
model = FlightDelayPredictor(input_dim).to(device)

batch_size = 256
learning_rate = 0.001
num_epochs = 300

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
)
early_stopping = EarlyStopping(patience=25, min_delta=1.0)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")


# %%
def train_model():
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None

    epoch_pbar = tqdm(range(num_epochs), desc="Training")

    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{avg_train_loss:.1f}",
                "val_loss": f"{avg_val_loss:.1f}",
                "lr": f"{current_lr:.6f}",
            }
        )

        if early_stopping(avg_val_loss):
            epoch_pbar.set_description(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


start_time = datetime.now()
train_losses, val_losses = train_model()
training_time = datetime.now() - start_time

# %%
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
if len(train_losses) > 10:
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
model.eval()
with torch.no_grad():
    train_predictions = model(X_train_tensor).cpu().numpy().flatten()
    test_predictions = model(X_test_tensor).cpu().numpy().flatten()

train_mae = mean_absolute_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("=" * 60)
print("PYTORCH NEURAL NETWORK RESULTS")
print("=" * 60)
print(f"Training - MAE: {train_mae:.2f} min, R²: {train_r2:.4f}")
print(f"Test     - MAE: {test_mae:.2f} min, R²: {test_r2:.4f}")
print(f"Training time: {training_time}")

# %%
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_train, train_predictions, alpha=0.5, s=1)
min_val, max_val = float(y_train.min()), float(y_train.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
plt.xlabel("Actual Delay (minutes)")
plt.ylabel("Predicted Delay (minutes)")
plt.title(f"Training: R² = {train_r2:.4f}")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(y_test, test_predictions, alpha=0.5, s=1)
min_val, max_val = float(y_test.min()), float(y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
plt.xlabel("Actual Delay (minutes)")
plt.ylabel("Predicted Delay (minutes)")
plt.title(f"Test: R² = {test_r2:.4f}")
plt.grid(True, alpha=0.3)

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
def calculate_feature_importance(model, X_tensor, y_tensor, feature_names):
    model.eval()
    with torch.no_grad():
        baseline_pred = model(X_tensor)
        baseline_mae = mean_absolute_error(
            y_tensor.cpu().numpy(), baseline_pred.cpu().numpy()
        )

    importances = []
    for i, feature_name in enumerate(tqdm(feature_names, desc="Feature importance")):
        X_permuted = X_tensor.clone()
        perm_idx = torch.randperm(X_permuted.shape[0])
        X_permuted[:, i] = X_permuted[perm_idx, i]

        with torch.no_grad():
            permuted_pred = model(X_permuted)
            permuted_mae = mean_absolute_error(
                y_tensor.cpu().numpy(), permuted_pred.cpu().numpy()
            )

        importances.append(permuted_mae - baseline_mae)

    return np.array(importances)


feature_importance = calculate_feature_importance(
    model, X_test_tensor, y_test_tensor, feature_columns
)
feature_importance_df = pd.DataFrame(
    {"feature": feature_columns, "importance": feature_importance}
).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 8))
top_features = feature_importance_df.head(15)
sns.barplot(data=top_features, x="importance", y="feature")
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance (MAE Increase)")
plt.tight_layout()
show_or_save_plot("feature_importance.png")

print("Top 10 most important features:")
print(feature_importance_df.head(10))

# %%
model_checkpoint = {
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
}

buf = io.BytesIO()
torch.save(model_checkpoint, buf)
buf.seek(0)
storage.s3.upload_fileobj(buf, config.bucket_name, f"{config.results_path}/model/pytorch_flight_delay_model.pth")

for name, obj in [
    ("feature_scaler.pkl", scaler),
    ("origin_encoder.pkl", le_origin),
    ("dest_encoder.pkl", le_dest),
    ("feature_imputer.pkl", imputer)
]:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    storage.s3.upload_fileobj(buf, config.bucket_name, f"{config.results_path}/model/{name}")

print(f"\nModel saved to s3://{config.bucket_name}/{config.results_path}/model/")


# %%
def predict_flight_delay(origin, dest, weather_data):
    predict_storage = Storage()
    
    buf = io.BytesIO()
    predict_storage.s3.download_fileobj(config.bucket_name, f"{config.results_path}/model/pytorch_flight_delay_model.pth", buf)
    buf.seek(0)
    checkpoint = torch.load(buf, map_location=device)

    model_arch = checkpoint["model_architecture"]
    loaded_model = FlightDelayPredictor(
        model_arch["input_dim"], model_arch["hidden_dims"], model_arch["dropout_rate"]
    ).to(device)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()

    artifacts = {}
    for name in ["feature_scaler.pkl", "origin_encoder.pkl", "dest_encoder.pkl", "feature_imputer.pkl"]:
        buf = io.BytesIO()
        predict_storage.s3.download_fileobj(config.bucket_name, f"{config.results_path}/model/{name}", buf)
        buf.seek(0)
        artifacts[name] = joblib.load(buf)

    scaler = artifacts["feature_scaler.pkl"]
    le_origin = artifacts["origin_encoder.pkl"]
    le_dest = artifacts["dest_encoder.pkl"]
    imputer = artifacts["feature_imputer.pkl"]

    input_data = weather_data.copy()

    # Add engineered features
    input_data["temp_diff"] = abs(
        input_data.get("dep_temp", 0) - input_data.get("arr_temp", 0)
    )
    input_data["total_prcp"] = input_data.get("dep_prcp", 0) + input_data.get(
        "arr_prcp", 0
    )
    input_data["dep_wind_stress"] = input_data.get("dep_wspd", 0) * (
        input_data.get("dep_coco", 1) / 10.0
    )
    input_data["arr_wind_stress"] = input_data.get("arr_wspd", 0) * (
        input_data.get("arr_coco", 1) / 10.0
    )
    input_data["pressure_diff"] = abs(
        input_data.get("dep_pres", 1013) - input_data.get("arr_pres", 1013)
    )
    input_data["dep_bad_weather"] = int(
        (input_data.get("dep_prcp", 0) > 5)
        or (input_data.get("dep_wspd", 0) > 30)
        or (input_data.get("dep_coco", 1) >= 6)
    )
    input_data["arr_bad_weather"] = int(
        (input_data.get("arr_prcp", 0) > 5)
        or (input_data.get("arr_wspd", 0) > 30)
        or (input_data.get("arr_coco", 1) >= 6)
    )

    try:
        input_data["Origin_encoded"] = le_origin.transform([origin])[0]
    except ValueError:
        input_data["Origin_encoded"] = -1

    try:
        input_data["Dest_encoded"] = le_dest.transform([dest])[0]
    except ValueError:
        input_data["Dest_encoded"] = -1

    feature_cols = checkpoint["feature_columns"]
    input_df = pd.DataFrame([input_data])

    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_cols]
    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    input_scaled = scaler.transform(input_imputed)
    input_tensor = torch.FloatTensor(input_scaled).to(device)

    with torch.no_grad():
        prediction = loaded_model(input_tensor).cpu().numpy()[0][0]

    return prediction


# %%
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

test_scenarios = [
    {"name": "Clear Weather", "weather": test_weather_data},
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

print("\n" + "=" * 60)
print("TESTING PREDICTION FUNCTION")
print("=" * 60)

for scenario in test_scenarios:
    try:
        prediction = predict_flight_delay("DEN", "DAY", scenario["weather"])
        delay_category = (
            "Early arrival"
            if prediction < 0
            else "On time"
            if prediction < 15
            else "Moderate delay"
            if prediction < 60
            else "Significant delay"
        )
        print(f"{scenario['name']}: {prediction:.1f} minutes → {delay_category}")
    except Exception as e:
        print(f"{scenario['name']}: ERROR - {e}")

print(f"\n{'=' * 60}")
print(f"Final Performance: R² = {test_r2:.4f}, MAE = {test_mae:.2f} min")
print(f"Training time: {training_time}, Device: {device}")
print("=" * 60)

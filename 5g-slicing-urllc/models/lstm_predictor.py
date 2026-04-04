"""
LSTM-based short-term latency predictor for URLLC slice.
Input:  sliding window of 24 historical steps (raw latency + rolling mean/std)
Output: predicted latency for next 6 steps (horizon = 6)

Only requires a 'latency_ms' column — no seq/recv/lost columns needed.

Usage:
    python3 lstm_predictor.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import random


# ── Hyperparameters ────────────────────────────────────────────────────────────
WINDOW_SIZE   = 24      # Past 24 steps (look-back window)
HORIZON       = 6       # Predict 6 steps ahead
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
EPOCHS        = 100
BATCH_SIZE    = 32
LR            = 0.001
TEST_RATIO    = 0.2
VAL_RATIO     = 0.1
PATIENCE      = 15
SMOOTH_WINDOW = 3       # Rolling-mean applied to target before training
SEED          = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Features derived from latency_ms only:
#   [raw_latency, rolling_mean_3, rolling_std_3]
N_FEATURES = 3


# ── Model ──────────────────────────────────────────────────────────────────────
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, output_size=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


# ── Helpers ────────────────────────────────────────────────────────────────────
def build_features(latency: np.ndarray) -> np.ndarray:
    """
    Build a (T, 3) feature matrix from a 1-D latency array.
    Features:
      0 – raw latency value
      1 – 3-step rolling mean
      2 – 3-step rolling std  (0-filled at edges)
    No external columns required.
    """
    s = pd.Series(latency, dtype=np.float32)
    roll_mean = s.rolling(3, min_periods=1).mean().values
    roll_std  = s.rolling(3, min_periods=1).std().fillna(0).values
    return np.stack([latency, roll_mean, roll_std], axis=1).astype(np.float32)


def create_sequences(features: np.ndarray, target: np.ndarray,
                     window: int, horizon: int):
    """
    features : (T, F)
    target   : (T,)
    returns  : X (N, window, F),  y (N, horizon)
    """
    X, y = [], []
    for i in range(len(target) - window - horizon + 1):
        X.append(features[i : i + window])
        y.append(target[i + window : i + window + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Main ───────────────────────────────────────────────────────────────────────
def train_and_evaluate():
    set_seed(SEED)

    # ── Load & validate ──────────────────────────────────────────────────────
    data_path = os.path.join(BASE_DIR, '..', 'data', 'urllc_timeseries.csv')
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    if 'latency_ms' not in df.columns:
        raise ValueError(
            f"CSV must contain a 'latency_ms' column. "
            f"Found: {list(df.columns)}"
        )

    df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
    df.dropna(subset=['latency_ms'], inplace=True)

    # Sort by seq if present (preserves temporal order), else keep file order
    if 'seq' in df.columns:
        df = df.sort_values('seq').reset_index(drop=True)

    latency = df['latency_ms'].values.astype(np.float32)

    print(f"Total samples    : {len(latency)}")
    print(f"Latency range    : {latency.min():.2f} – {latency.max():.2f} ms")
    print(f"Latency mean     : {latency.mean():.2f} ms")
    print(f"Window / Horizon : {WINDOW_SIZE} / {HORIZON}")

    # ── Feature engineering (latency-only) ──────────────────────────────────
    features = build_features(latency)          # (T, 3)

    # Smooth target to reduce pure measurement noise
    raw_target    = latency.copy()
    smooth_target = (pd.Series(raw_target)
                       .rolling(SMOOTH_WINDOW, min_periods=1, center=False)
                       .mean()
                       .values
                       .astype(np.float32))

    # ── Sliding-window sequences ─────────────────────────────────────────────
    X, y = create_sequences(features, smooth_target, WINDOW_SIZE, HORIZON)

    n         = len(X)
    train_end = int((1.0 - TEST_RATIO - VAL_RATIO) * n)
    val_end   = int((1.0 - TEST_RATIO) * n)

    X_train, y_train = X[:train_end],        y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    print(f"Samples – train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

    # ── Scaling ──────────────────────────────────────────────────────────────
    feat_scaler = MinMaxScaler()
    feat_scaler.fit(X_train.reshape(-1, N_FEATURES))

    tgt_scaler = MinMaxScaler()
    tgt_scaler.fit(y_train.reshape(-1, 1))

    def scale_feat(arr):    # (N, W, F) → (N, W, F)
        s = arr.shape
        return feat_scaler.transform(arr.reshape(-1, N_FEATURES)).reshape(s)

    def scale_tgt(arr):     # (N, H) → (N, H)
        s = arr.shape
        return tgt_scaler.transform(arr.reshape(-1, 1)).reshape(s)

    X_train_s = scale_feat(X_train);  y_train_s = scale_tgt(y_train)
    X_val_s   = scale_feat(X_val);    y_val_s   = scale_tgt(y_val)
    X_test_s  = scale_feat(X_test)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    X_train_t = torch.FloatTensor(X_train_s)
    y_train_t = torch.FloatTensor(y_train_s)
    X_val_t   = torch.FloatTensor(X_val_s)
    y_val_t   = torch.FloatTensor(y_val_s)
    X_test_t  = torch.FloatTensor(X_test_s)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ── Model / optimiser ────────────────────────────────────────────────────
    model     = TrafficLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, HORIZON)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5, min_lr=1e-5
    )

    train_losses, val_losses = [], []
    best_val_loss  = float('inf')
    best_state     = None
    patience_count = 0

    # ── Training loop ────────────────────────────────────────────────────────
    print("\nTraining:")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                f"Train Loss: {avg_train:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds_s = model(X_test_t).numpy()

    preds  = tgt_scaler.inverse_transform(preds_s.reshape(-1, 1)).reshape(preds_s.shape)
    actual = y_test   # already in ms

    mae  = mean_absolute_error(actual.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(actual.flatten(), preds.flatten()))

    print(f"\nResults:")
    print(f"MAE:  {mae:.4f} ms")
    print(f"RMSE: {rmse:.4f} ms")

    # ── Save model ───────────────────────────────────────────────────────────
    save_dir = os.path.join(BASE_DIR,"..", 'saved')
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, 'lstm_urllc.pth'))

    # Metrics
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'window': WINDOW_SIZE,
        'horizon': HORIZON,
        'epochs': EPOCHS,
        'hidden_size': HIDDEN_SIZE,
        'test_ratio': TEST_RATIO,
        'val_ratio': VAL_RATIO
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Scaler params as .npy (feat scaler + target scaler)
    np.save(os.path.join(save_dir, 'feat_scaler_min.npy'),   feat_scaler.min_)
    np.save(os.path.join(save_dir, 'feat_scaler_scale.npy'), feat_scaler.scale_)
    np.save(os.path.join(save_dir, 'tgt_scaler_min.npy'),    tgt_scaler.min_)
    np.save(os.path.join(save_dir, 'tgt_scaler_scale.npy'),  tgt_scaler.scale_)

    # ── Spike detection ──────────────────────────────────────────────────────
    threshold   = raw_target.mean() + 2.0 * raw_target.std()
    spike_count = 0
    print(f"\nSpike threshold : {threshold:.2f} ms")
    for i, pred_seq in enumerate(preds[:20]):
        if pred_seq.max() > threshold:
            print(f"  Spike predicted at sample {i}: {pred_seq.max():.2f} ms")
            spike_count += 1
    if spike_count == 0:
        print("  No spikes detected in first 20 test samples.")

    # ── Plot ─────────────────────────────────────────────────────────────────
    viz_dir = os.path.join(BASE_DIR, '..', 'visualization')
    os.makedirs(viz_dir, exist_ok=True)

    n_plot = min(100, len(preds))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses,   label='Val')
    axes[0].set_title('Training / Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(actual[:n_plot, 0], label='Actual',    alpha=0.8)
    axes[1].plot(preds[:n_plot, 0],  label='Predicted', alpha=0.8)
    axes[1].set_title(f'Prediction vs Actual (first {n_plot} test samples, step-1)')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(viz_dir, 'lstm_results.png')
    plt.savefig(out_png, dpi=150)
    print(f"\nPlot saved → {out_png}")

    return model, feat_scaler, tgt_scaler


if __name__ == '__main__':
    train_and_evaluate()
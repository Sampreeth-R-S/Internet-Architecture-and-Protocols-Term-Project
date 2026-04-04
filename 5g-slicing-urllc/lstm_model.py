"""
URLLC Latency Prediction using LSTM
Fixed: multivariate features, direct prediction, proper scaling, early stopping
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
import random
import json


# Parameters
WINDOW_SIZE = 10
HORIZON = 3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
TEST_RATIO = 0.2
VAL_RATIO = 0.1
PATIENCE = 15
SMOOTH_WINDOW = 3   # rolling-mean steps applied to target before training
SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

N_FEATURES = 4   # max_lat, mean_lat, std_lat, loss_rate


class TrafficLSTM(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=128, num_layers=2, output_size=HORIZON):
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


def create_sequences_mv(features, target, window, horizon):
    """features: (T, F), target: (T,)  →  X: (N, window, F), y: (N, horizon)"""
    X, y = [], []
    for i in range(len(target) - window - horizon + 1):
        X.append(features[i:i + window])
        y.append(target[i + window:i + window + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_and_evaluate():
    set_seed(SEED)

    # Load data
    data_path = os.path.join(BASE_DIR, 'data', 'Training_data.csv')
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
    df.dropna(subset=['latency_ms'], inplace=True)

    # Build one row per seq with multiple features
    agg = (df.groupby('seq')
             .agg(
                 max_lat=('latency_ms', 'max'),
                 mean_lat=('latency_ms', 'mean'),
                 std_lat=('latency_ms', 'std'),
                 total_recv=('recv', 'sum'),
                 total_lost=('lost', 'sum'),
             )
             .fillna(0)
             .sort_index()
             .reset_index())

    agg['loss_rate'] = agg['total_lost'] / (agg['total_recv'] + 1e-6)

    # Features fed into LSTM at each time step
    feat_cols = ['max_lat', 'mean_lat', 'std_lat', 'loss_rate']
    features = agg[feat_cols].values.astype(np.float32)  # (T, 4)

    # Target: max latency smoothed to reduce pure measurement noise
    raw_target = agg['max_lat'].values.astype(np.float32)
    smooth_target = (pd.Series(raw_target)
                       .rolling(SMOOTH_WINDOW, min_periods=1, center=False)
                       .mean()
                       .values
                       .astype(np.float32))

    T = len(smooth_target)
    print(f"Time steps: {T}")
    print(f"Latency range: {raw_target.min():.2f} - {raw_target.max():.2f} ms  |  mean: {raw_target.mean():.2f} ms")

    # Build sliding-window sequences
    X, y = create_sequences_mv(features, smooth_target, WINDOW_SIZE, HORIZON)

    # Train / val / test split (chronological)
    n = len(X)
    train_end = int((1.0 - TEST_RATIO - VAL_RATIO) * n)
    val_end   = int((1.0 - TEST_RATIO) * n)

    X_train, y_train = X[:train_end],       y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    print(f"Samples — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

    # Separate scalers: one for features (per-column), one for target
    feat_scaler = MinMaxScaler()
    feat_scaler.fit(X_train.reshape(-1, N_FEATURES))

    tgt_scaler = MinMaxScaler()
    tgt_scaler.fit(y_train.reshape(-1, 1))

    def scale_feat(arr):   # arr: (N, W, F)
        s = arr.shape
        return feat_scaler.transform(arr.reshape(-1, N_FEATURES)).reshape(s)

    def scale_tgt(arr):    # arr: (N, H)
        s = arr.shape
        return tgt_scaler.transform(arr.reshape(-1, 1)).reshape(s)

    X_train_s = scale_feat(X_train)
    y_train_s = scale_tgt(y_train)
    X_val_s   = scale_feat(X_val)
    y_val_s   = scale_tgt(y_val)
    X_test_s  = scale_feat(X_test)

    # Tensors — X shape: (samples, window, N_FEATURES)
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

    print("\nTraining...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Train: {avg_train:.6f} | Val: {val_loss:.6f}")

        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        preds_s = model(X_test_t).numpy()

    # Inverse-transform back to ms
    preds  = tgt_scaler.inverse_transform(preds_s.reshape(-1, 1)).reshape(preds_s.shape)
    actual = y_test  # already in ms (never scaled)

    mae  = mean_absolute_error(actual.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(actual.flatten(), preds.flatten()))

    print(f"\nTest MAE:  {mae:.4f} ms")
    print(f"Test RMSE: {rmse:.4f} ms")

    # Save model and metrics
    save_dir = os.path.join(BASE_DIR, 'saved')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'lstm_urllc.pth'))
    with open(os.path.join(save_dir, 'metrics_urllc.json'), 'w') as f:
        json.dump({'MAE': float(mae), 'RMSE': float(rmse)}, f, indent=2)

    # Spike detection
    threshold = raw_target.mean() + 2.0 * raw_target.std()
    print(f"\nSpike threshold: {threshold:.2f} ms")
    for i, pred_seq in enumerate(preds[:20]):
        if pred_seq.max() > threshold:
            print(f"  Spike predicted at sample {i}: {pred_seq.max():.2f} ms")

    # Plot
    viz_dir = os.path.join(BASE_DIR, 'visualization')
    os.makedirs(viz_dir, exist_ok=True)

    n_plot = min(100, len(preds))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(train_losses, label='Train loss')
    axes[0].plot(val_losses,   label='Val loss')
    axes[0].set_title('Training / Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(actual[:n_plot, 0], label='Actual',    linewidth=1.5)
    axes[1].plot(preds[:n_plot, 0],  label='Predicted', linewidth=1.5, linestyle='--')
    axes[1].set_title('URLLC Latency Prediction (1-step ahead)')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_xlabel('Time step')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'lstm_urllc_final.png'), dpi=150)
    print("\nPlot saved to visualization/lstm_urllc_final.png")

    return model


if __name__ == '__main__':
    train_and_evaluate()

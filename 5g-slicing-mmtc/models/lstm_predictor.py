"""
LSTM-based congestion predictor for mMTC slice.
Predicts future device connection density and packet rate to detect
congestion peaks before they overwhelm the mMTC slice.

Input:  sliding window of historical packet_rate (24 steps = 2 min at 5s intervals)
Output: predicted packet_rate for next 6 steps (30 seconds)

When predicted packet_rate exceeds the slice capacity threshold,
the controller proactively expands connection capacity.

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

WINDOW_SIZE = 60      # Past 60 steps (5 minutes) — captures full IoT reporting cycle
HORIZON = 6           # Predict 6 steps ahead (30 seconds)
HIDDEN_SIZE = 128
NUM_LAYERS = 3
EPOCHS = 300
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 1e-4
TEST_RATIO = 0.15
SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


def create_sequences(data, window, horizon):
    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i:i + window])
        y.append(data[i + window:i + window + horizon])
    return np.array(X), np.array(y)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scale_2d(array, scaler):
    return scaler.transform(array.reshape(-1, 1)).reshape(array.shape)


def train_and_evaluate():
    set_seed(SEED)

    # Load data
    data_path = os.path.join(BASE_DIR, '..', 'data', 'lstm_controller_stats.csv')
    df = pd.read_csv(data_path)

    # mMTC key metric: packet_rate (connection density proxy)
    values = df['packet_rate'].values.astype(np.float32)

    # Create sequences
    X, y = create_sequences(values, WINDOW_SIZE, HORIZON)

    # Chronological split
    n_samples = len(X)
    train_end = int((1.0 - TEST_RATIO) * n_samples)

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]

    # Fit scaler only on train partition
    scaler = MinMaxScaler()
    scaler.fit(X_train.reshape(-1, 1))

    X_train = scale_2d(X_train, scaler)
    X_test = scale_2d(X_test, scaler)
    y_train = scale_2d(y_train, scaler)
    y_test = scale_2d(y_test, scaler)

    # To tensors
    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = TrafficLSTM(1, HIDDEN_SIZE, NUM_LAYERS, HORIZON)
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Train
    losses = []

    print("Training mMTC LSTM Congestion Predictor:")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.6f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).numpy()
        train_preds = model(X_train_t).numpy()

    # Inverse scaling
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    train_preds_inv = scaler.inverse_transform(train_preds.reshape(-1, 1)).reshape(train_preds.shape)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)

    mae = mean_absolute_error(y_test_inv.flatten(), preds_inv.flatten())
    rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), preds_inv.flatten()))

    print(f"\nResults:")
    print(f"MAE:  {mae:.4f} pps (packets per second)")
    print(f"RMSE: {rmse:.4f} pps")

    # Save model
    save_dir = os.path.join(BASE_DIR, 'saved')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'lstm_mmtc.pth'))

    # Save scaler params and metrics
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'window': WINDOW_SIZE,
        'horizon': HORIZON,
        'epochs': EPOCHS,
        'hidden_size': HIDDEN_SIZE,
        'test_ratio': TEST_RATIO,
        'prediction_target': 'packet_rate',
        'slice_type': 'mMTC'
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(save_dir, 'scaler_min.npy'), scaler.min_)
    np.save(os.path.join(save_dir, 'scaler_scale.npy'), scaler.scale_)

    # Plot
    viz_dir = os.path.join(BASE_DIR, '..', 'visualization')
    os.makedirs(viz_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('mMTC LSTM Congestion Predictor', fontsize=14, fontweight='bold')

    axes[0].plot(losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('SmoothL1 Loss')
    axes[0].legend(['Train'])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(y_train_inv[:100, 0], label='Actual', alpha=0.8)
    axes[1].plot(train_preds_inv[:100, 0], label='Predicted', alpha=0.8)
    axes[1].set_title('Prediction vs Actual (first 100 samples, step-1)')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Packet Rate (pps)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(viz_dir, 'lstm_results.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved → {plot_path}")

    return model, scaler


if __name__ == '__main__':
    train_and_evaluate()

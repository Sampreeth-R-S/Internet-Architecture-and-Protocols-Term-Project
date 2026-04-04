"""
Generates synthetic eMBB traffic time-series data mimicking
4K video streaming patterns with bursty behavior.

Usage:
    python3 generate_synthetic_data.py
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 8640  # 12 hours at 5-second intervals

# Base load: sinusoidal daily pattern (peak at hours 10, 14, 20)
t = np.arange(N)
hour_of_day = (t * 5 / 3600) % 24
base = 30 + 20 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 3)

# Bursty peaks (simulating 4K streaming surges)
bursts = np.zeros(N)
for _ in range(40):
    start = np.random.randint(0, N - 120)
    duration = np.random.randint(30, 120)
    intensity = np.random.uniform(20, 50)
    bursts[start:start + duration] += intensity

# Random noise
noise = np.random.normal(0, 3, N)

throughput = np.clip(base + bursts + noise, 0, 100)
packet_rate = throughput * np.random.uniform(80, 120, N)
active_ues = np.clip(
    5 + throughput / 10 + np.random.normal(0, 1, N), 1, 20
).astype(int)

df = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-01', periods=N, freq='5s'),
    'throughput_mbps': np.round(throughput, 2),
    'packet_rate': np.round(packet_rate, 0).astype(int),
    'active_ues': active_ues
})

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, 'embb_traffic_timeseries.csv')
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} samples → {output_path}")
print(df.describe())

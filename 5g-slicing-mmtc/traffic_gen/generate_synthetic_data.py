"""
Generates synthetic mMTC traffic time-series data mimicking
massive IoT sensor patterns with periodic reports and event bursts.

mMTC characteristics vs eMBB:
  - Low throughput per device (kbps range, not Mbps)
  - Very high device/packet count (thousands of sensors)
  - Small packets (48-128 bytes)
  - Periodic + event-driven bursts
  - Aggregate throughput in low Mbps range

Usage:
    python3 generate_synthetic_data.py
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 8640  # 12 hours at 5-second intervals

t = np.arange(N)
hour_of_day = (t * 5 / 3600) % 24

# --- Device count: sinusoidal pattern peaking during business hours ---
# mMTC devices connect/disconnect; base ~800, peak ~1200
device_count = 800 + 200 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
device_count += np.random.normal(0, 30, N)
device_count = np.clip(device_count, 200, 1500).astype(int)

# --- Packet rate: driven by device count and reporting intervals ---
# Each device sends ~1 packet per 30s average => ~device_count/30 pps per tick (5s)
# Plus jitter and periodic spikes
base_packet_rate = device_count * (5 / 30)  # packets per 5s tick
# Add periodic reporting spikes (e.g., every 5 min all meters report)
periodic_spikes = np.zeros(N)
for i in range(0, N, 60):  # every 5 minutes (60 * 5s = 300s)
    spike_width = np.random.randint(3, 8)
    spike_intensity = np.random.uniform(1.5, 3.0)
    end = min(i + spike_width, N)
    periodic_spikes[i:end] = spike_intensity

# Event-driven bursts (alarm triggers)
bursts = np.zeros(N)
for _ in range(25):  # 25 burst events in 12 hours
    start = np.random.randint(0, N - 60)
    duration = np.random.randint(10, 60)
    intensity = np.random.uniform(2.0, 5.0)
    bursts[start:start + duration] = intensity

packet_rate = base_packet_rate * (1 + periodic_spikes + bursts)
packet_rate += np.random.normal(0, 5, N)
packet_rate = np.clip(packet_rate, 0, 2000)

# --- Throughput: small packets, low bandwidth ---
# Average packet size ~80 bytes => throughput = packet_rate * 80 * 8 / 1e6 Mbps
avg_packet_size = 80 + 20 * np.random.randn(N)  # bytes
avg_packet_size = np.clip(avg_packet_size, 48, 128)
throughput_mbps = packet_rate * avg_packet_size * 8 / 1e6
throughput_mbps = np.clip(throughput_mbps, 0, 10)  # mMTC cap at ~10 Mbps aggregate

# --- Active devices per tick ---
active_devices = np.clip(
    device_count * (0.05 + 0.03 * (periodic_spikes + bursts)),
    1, device_count
).astype(int)

df = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-01', periods=N, freq='5s'),
    'throughput_mbps': np.round(throughput_mbps, 4),
    'packet_rate': np.round(packet_rate, 0).astype(int),
    'active_devices': active_devices,
    'device_count': device_count,
    'avg_packet_size_bytes': np.round(avg_packet_size, 0).astype(int),
})

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, 'mmtc_traffic_timeseries.csv')
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} samples → {output_path}")
print(df.describe())

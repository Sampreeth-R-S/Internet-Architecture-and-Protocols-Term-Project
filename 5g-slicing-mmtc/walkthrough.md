# 5G mMTC Network Slicing - Complete Command Guide

> Run these commands **in order**. All project source files have already been created.
>
> **mMTC (massive Machine Type Communications)** handles massive IoT sensor
> connectivity — thousands of devices sending small UDP packets. This contrasts
> with the eMBB slice which handles high-bandwidth video streaming.

---

## Phase 1: Install Dependencies & Configure mMTC Slice

### Step 1.1 - Install Node.js 20.x

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

Verify: `node --version`, should show `v20.x.x`

---

### Step 1.2 - Install build tools

```bash
sudo apt install -y cmake libsctp-dev lksctp-tools git
```

Verify:
```bash
cmake --version
```

---

### Step 1.3 - Install Open5GS WebUI

```bash
curl -fsSL https://open5gs.org/open5gs/assets/webui/install | sudo -E bash -
```

Verify: Open `http://localhost:9999` in a browser. Login: **admin** / **1423**

---

### Step 1.4 - Configure mMTC Slice in Open5GS

<<<<<<< HEAD
You need to edit **two** config files to add the mMTC slice (`sst: 3`, `sd: 000003`).
=======
You need to edit **two** config files to add the mMTC slice (`sd: 000002`).
>>>>>>> f3c1c30ef8652d3134ab471d44772c65f5fbd4fb
If the eMBB slice (`sd: 000001`) is already configured, **add** the mMTC
slice alongside it.

#### Edit AMF config:

```bash
sudo nano /etc/open5gs/amf.yaml
```

Find the `plmn_support` section and ensure **both** slices are listed:

```yaml
  plmn_support:
    - plmn_id:
        mcc: 999
        mnc: 70
      s_nssai:
        - sst: 1
          sd: 000001    # eMBB slice
<<<<<<< HEAD
        - sst: 3
          sd: 000003    # mMTC slice
=======
        - sst: 1
          sd: 000002    # mMTC slice
>>>>>>> f3c1c30ef8652d3134ab471d44772c65f5fbd4fb
```

#### Edit NSSF config:

```bash
sudo nano /etc/open5gs/nssf.yaml
```

Add the mMTC slice entry alongside eMBB:

```yaml
      nsi:
        - uri: http://127.0.0.10:7777
          s_nssai:
            sst: 1
            sd: 000001    # eMBB
        - uri: http://127.0.0.10:7777
          s_nssai:
<<<<<<< HEAD
            sst: 3
            sd: 000003    # mMTC
=======
            sst: 1
            sd: 000002    # mMTC
>>>>>>> f3c1c30ef8652d3134ab471d44772c65f5fbd4fb
```

#### Restart affected services:

```bash
sudo systemctl restart open5gs-amfd open5gs-nssfd
```

Verify both are running:
```bash
sudo systemctl status open5gs-amfd open5gs-nssfd
```

---

### Step 1.5 - Register mMTC Subscriber via WebUI

1. Open `http://localhost:9999` (login: **admin** / **1423**)
2. Click the **"+"** button to add a subscriber
3. Fill in these fields:
   - **IMSI:** `999700000000011`
   - **Key (K):** `465B5CE8B199B49FAA5F0A2EE238A6BC`
   - **OPC:** `E8ED289DEBA952E4283B54E88E6183CA`
4. Scroll down to the **Slice Configuration** section:
<<<<<<< HEAD
   - **SST:** `3`
   - **SD:** `000003`
=======
   - **SST:** `1`
   - **SD:** `000002`
>>>>>>> f3c1c30ef8652d3134ab471d44772c65f5fbd4fb
   - **DNN:** `internet`
   - **Session-AMBR Downlink:** `10` Mbps *(much lower than eMBB — IoT sensors don't need high bandwidth)*
   - **Session-AMBR Uplink:** `5` Mbps
5. Click **Save**

> **Note:** The mMTC slice uses much lower AMBR values than eMBB (10/5 Mbps vs 100/50 Mbps)
> because IoT sensors send small packets (48-128 bytes) rather than streaming 4K video.

---

### Step 1.6 - Build UERANSIM (if not already built)

```bash
cd ~/Desktop/5g-slicing-mmtc
git clone https://github.com/aligungr/UERANSIM.git ueransim-src
cd ueransim-src
make
```

Verify: `ls build/nr-gnb build/nr-ue`, both binaries should exist

> If you already built UERANSIM for the eMBB slice, you can symlink it:
> ```bash
> ln -s ~/Desktop/5g-slicing-embb/ueransim-src ~/Desktop/5g-slicing-mmtc/ueransim-src
> ```

---

### Step 1.7 - Start UERANSIM & Test mMTC Connectivity

**You need 3 separate terminals for this:**

**Terminal 1 - Start mMTC gNB:**
```bash
cd ~/Desktop/5g-slicing-mmtc/ueransim-src
sudo ./build/nr-gnb -c ../ueransim/gnb-mmtc.yaml
```
Expected: `NG Setup procedure is successful`

**Terminal 2 - Start mMTC UE:**
```bash
cd ~/Desktop/5g-slicing-mmtc/ueransim-src
sudo ./build/nr-ue -c ../ueransim/ue-mmtc.yaml
```
Expected: `PDU Session establishment is successful`

**Terminal 3 - Test connectivity:**
```bash
# Check the tunnel interface was created
ip addr show uesimtun0

# Ping through the 5G tunnel
ping -I uesimtun0 10.45.0.1 -c 5
```

---

## Phase 2: Install Python Dependencies & Generate Data

### Step 2.1 - Install Python ML packages

```bash
pip3 install numpy pandas torch scikit-learn matplotlib
```

<<<<<<< HEAD
### Step 2.2 - Generate mMTC training data

**You need 2 separate terminals for this:**

**Terminal A - Start data collector:**
```bash
cd ~/Desktop/5g-slicing-mmtc/traffic_gen
python3 collect_metrics.py
```

**Terminal B - Start mMTC sensor traffic:**
```bash
cd ~/Desktop/5g-slicing-mmtc/traffic_gen
python3 mmtc_sensor_traffic.py
```
=======
### Step 2.2 - Generate synthetic mMTC training dataset

```bash
cd ~/Desktop/5g-slicing-mmtc/traffic_gen
python3 generate_synthetic_data.py
```

This generates an mMTC traffic time-series with:
- **Thousands of IoT devices** sending small packets periodically
- **Event-driven bursts** (alarm sensors triggering simultaneously)
- **Periodic reporting spikes** (all smart meters reporting at once)
- Low throughput (< 10 Mbps aggregate) but high packet rates (100-1500+ pps)
>>>>>>> f3c1c30ef8652d3134ab471d44772c65f5fbd4fb

Verify:
```bash
wc -l ../data/mmtc_traffic_timeseries.csv
<<<<<<< HEAD
=======
# Should show 8641 lines (8640 data + header)
```

---

### Step 2.3 - (Optional) Run the live mMTC sensor simulator

This script simulates 1000 IoT sensors sending small UDP packets:

```bash
cd ~/Desktop/5g-slicing-mmtc/traffic_gen
python3 mmtc_sensor_traffic.py --sensors 1000 --duration 300
```

In live mode (sends real UDP packets through the 5G tunnel):
```bash
sudo python3 mmtc_sensor_traffic.py --sensors 100000 --duration 1200 --live --interface uesimtun0 --target 10.45.0.1 --port 9999 --loss-rate 0.02 --max-retries 2 --retry-delay-ms 25 --sleep-ratio 0.7 --sleep-period-s 300 --sleep-window-s 120 --seed 42
>>>>>>> f3c1c30ef8652d3134ab471d44772c65f5fbd4fb
```

---

## Phase 3: Train the LSTM Model

```bash
cd ~/Desktop/5g-slicing-mmtc/models
python3 lstm_predictor.py
```

Expected output:
- Training progress every 10 epochs
- Model saved to `saved/lstm_mmtc.pth`
- Plot saved to `../visualization/lstm_results.png`

The mMTC LSTM predicts **packet rate** (connection density proxy) instead of
throughput. When it predicts a surge in device connections, the controller
proactively expands the slice's connection capacity.

Verify:
```bash
ls saved/
# Should contain: lstm_mmtc.pth, metrics.json, scaler_min.npy, scaler_scale.npy
```

---
<<<<<<< HEAD
=======

## Phase 4: Run the Zero-Touch Controller

```bash
cd ~/Desktop/5g-slicing-mmtc/controller
python3 zero_touch_controller.py
```

Expected output:
- Prints EXPAND/CONTRACT decisions as it processes packet rate data
- Summary showing counts of expansions, contractions, holds
- Saves decision log to `../data/controller_decisions.json`

The mMTC controller manages:
- **Connection capacity** (max simultaneous device transmissions in pps)
- **Slice bandwidth** (low: 5-50 Mbps, since each device uses kbps)
- **5QI 79** (non-GBR, delay-tolerant IoT traffic)

Verify:
```bash
cat ../data/controller_decisions.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Total: {len(d)} decisions'); print({a: sum(1 for x in d if x['action']==a) for a in ['expand','contract','hold']})"
```

---

## Phase 5: Evaluation & Visualization

### Step 5.1 - Run baseline comparison

```bash
cd ~/Desktop/5g-slicing-mmtc/evaluation
python3 compare_baselines.py
```

Expected: Table comparing Static vs Reactive vs Proactive metrics including:
- Packet Delivery Ratio (PDR)
- Congestion rate
- SLA violation rate
- Resource waste

### Step 5.2 - Generate all visualization plots

```bash
cd ~/Desktop/5g-slicing-mmtc/visualization
python3 dashboard.py
```

Expected plots saved:
- `traffic_timeseries.png` — Packet rate, device count, and throughput over time
- `controller_decisions.png` — Decision timeline with expand/contract markers
- `baseline_comparison_dashboard.png` — Static vs Reactive vs Proactive bar charts
- `device_density_heatmap.png` — Hourly device density patterns

---

## Optional: Live Traffic Collection (requires UERANSIM running)

### Start UDP sensor simulation (Terminal A):
```bash
cd ~/Desktop/5g-slicing-mmtc/traffic_gen
python3 mmtc_sensor_traffic.py --sensors 2000 --duration 600 --live --target 10.45.0.1
```

### Start data collector (Terminal B):
```bash
cd ~/Desktop/5g-slicing-mmtc/traffic_gen
sudo python3 collect_metrics.py
```

### Run controller in live mode (Terminal C):
```bash
cd ~/Desktop/5g-slicing-mmtc/controller
python3 zero_touch_controller.py --live
```

---

## Key Differences: mMTC vs eMBB Slicing

| Aspect | eMBB Slice | mMTC Slice |
|---|---|---|
| **Traffic** | 4K video streaming (high BW) | IoT sensor data (small packets) |
| **Bandwidth** | 100-2500 Mbps per slice | 5-50 Mbps per slice |
| **Packet Size** | Large (1400+ bytes) | Small (48-128 bytes) |
| **Device Count** | 5 UEs | 1000+ sensors |
| **5QI** | 8/9 (GBR, real-time) | 79 (non-GBR, delay-tolerant) |
| **LSTM Predicts** | Throughput (Mbps) | Packet rate (pps) |
| **Controller Action** | Expand/contract bandwidth | Expand/contract connection capacity |
| **Traffic Gen** | iperf3 + curl (HTTP Range) | Custom Python UDP sender |
| **Session-AMBR DL** | 100 Mbps | 10 Mbps |
| **Session-AMBR UL** | 50 Mbps | 5 Mbps |

---
>>>>>>> f3c1c30ef8652d3134ab471d44772c65f5fbd4fb

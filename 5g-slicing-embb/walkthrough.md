# 5G eMBB Network Slicing - Complete Command Guide

> Run these commands **in order**. All project source files have already been created

---

## Phase 1: Install Dependencies & Configure eMBB Slice

### Step 1.1 - Install Node.js 20.x

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

Verify: `node --version`, should show `v20.x.x`

---

### Step 1.2 - Install iperf3, cmake, build tools

```bash
sudo apt install -y iperf3 cmake libsctp-dev lksctp-tools git
```

Verify:
```bash
iperf3 --version
cmake --version
```

---

### Step 1.3 - Install Open5GS WebUI

```bash
curl -fsSL https://open5gs.org/open5gs/assets/webui/install | sudo -E bash -
```

Verify: Open `http://localhost:9999` in a browser. Login: **admin** / **1423**

---

### Step 1.4 - Configure eMBB Slice in Open5GS

You need to edit **two** config files to add the Slice Differentiator (`sd: 000001`).

#### Edit AMF config:

```bash
sudo nano /etc/open5gs/amf.yaml
```

Find this section:
```yaml
  plmn_support:
    - plmn_id:
        mcc: 999
        mnc: 70
      s_nssai:
        - sst: 1
```

**Change it to:**
```yaml
  plmn_support:
    - plmn_id:
        mcc: 999
        mnc: 70
      s_nssai:
        - sst: 1
          sd: 000001
```

#### Edit NSSF config:

```bash
sudo nano /etc/open5gs/nssf.yaml
```

Find this section:
```yaml
      nsi:
        - uri: http://127.0.0.10:7777
          s_nssai:
            sst: 1
```

**Change it to:**
```yaml
      nsi:
        - uri: http://127.0.0.10:7777
          s_nssai:
            sst: 1
            sd: 000001
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

### Step 1.5 - Register eMBB Subscriber via WebUI

1. Open `http://localhost:9999` (login: **admin** / **1423**)
2. Click the **"+"** button to add a subscriber
3. Fill in these fields:
   - **IMSI:** `999700000000001`
   - **Key (K):** `465B5CE8B199B49FAA5F0A2EE238A6BC`
   - **OPC:** `E8ED289DEBA952E4283B54E88E6183CA`
4. Scroll down to the **Slice Configuration** section:
   - **SST:** `1`
   - **SD:** `000001`
   - **DNN:** `internet`
   - **Session-AMBR Downlink:** `100` Mbps
   - **Session-AMBR Uplink:** `50` Mbps
5. Click **Save**

---

### Step 1.6 - Build UERANSIM

```bash
cd ~/Desktop/5g-slicing-embb
git clone https://github.com/aligungr/UERANSIM.git ueransim-src
cd ueransim-src
make
```

Verify: `ls build/nr-gnb build/nr-ue`, both binaries should exist

---

### Step 1.7 - Start UERANSIM & Test Connectivity

**You need 3 separate terminals for this:**

**Terminal 1 - Start gNB:**
```bash
cd ~/Desktop/5g-slicing-embb/ueransim-src
sudo ./build/nr-gnb -c ../ueransim/gnb-embb.yaml
```
Expected: `NG Setup procedure is successful`

**Terminal 2 - Start UE:**
```bash
cd ~/Desktop/5g-slicing-embb/ueransim-src
sudo ./build/nr-ue -c ../ueransim/ue-embb.yaml
```
Expected: `PDU Session establishment is successful`

**Terminal 3 - Test connectivity:**
```bash
# Check the tunnel interface was created
ip addr show uesimtun0

# Ping through the 5G tunnel
ping -I uesimtun0 10.45.0.1 -c 5
```
Expected: Replies from `10.45.0.1` (No replies will be available if the firewall blocks ping requests, in that case just check if the server is online)

---

## Phase 2: Install Python Dependencies & Generate Data

### Step 2.1 - Install Python ML packages

```bash
pip3 install numpy pandas torch scikit-learn matplotlib
```

### Step 2.2 - Generate synthetic training dataset

```bash
cd ~/Desktop/5g-slicing-embb/traffic_gen
python collect_metrics.py
```
Then in a second terminal:
```bash
cd ~/Desktop/5g-slicing-embb/traffic_gen
chmod +x netflix_4k_streaming.sh
./netflix_4k_streaming.sh
```


Verify:
```bash
wc -l ../data/embb_traffic_timeseries.csv
```

---

## Phase 3: Train the LSTM Model

```bash
cd ~/Desktop/5g-slicing-embb/models
python3 lstm_predictor.py
```

Expected output:
- Training progress every 10 epochs
- Model saved to `saved/lstm_embb.pth`
- Plot saved to `../visualization/lstm_results.png`

Verify:
```bash
ls saved/
# Should contain: lstm_embb.pth
```

---

## Phase 4: Run the Zero-Touch Controller

```bash
cd ~/Desktop/5g-slicing-embb/controller
python3 zero_touch_controller.py
```

Expected output:
- Prints EXPAND/CONTRACT decisions as it processes the dataset
- Summary showing counts of expansions, contractions, holds
- Saves decision log to `../data/controller_decisions.json`

Verify:
```bash
cat ../data/controller_decisions.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Total: {len(d)} decisions'); print({a: sum(1 for x in d if x[\"action\"]==a) for a in ['expand','contract','hold']})"
```

---

## Phase 5: Evaluation & Visualization

### Step 5.1 - Run baseline comparison

```bash
cd ~/Desktop/5g-slicing-embb/evaluation
python3 compare_baselines.py
```

Expected: Table comparing Static vs Reactive vs Proactive metrics

### Step 5.2 - Generate all visualization plots

```bash
cd ~/Desktop/5g-slicing-embb/visualization
python3 dashboard.py
```

Expected: 4 plots saved:
- `traffic_timeseries.png`
- `controller_decisions.png`
- `baseline_comparison_dashboard.png`
- `slice_utilization_heatmap.png`

---

## Optional: Live Traffic Collection (requires UERANSIM running)

### Start iperf3 server:
```bash
# On the UPF machine (same machine in this case)
iperf3 -s -p 5201 &
```

### Start data collector (Terminal A):
```bash
cd ~/Desktop/5g-slicing-embb/traffic_gen
sudo python3 collect_metrics.py
```

### Start traffic generator (Terminal B):
```bash
cd ~/Desktop/5g-slicing-embb/traffic_gen
chmod +x netflix_4k_streaming.sh
sudo ./netflix_4k_streaming.sh
```

### Run controller in live mode (Terminal C):
```bash
cd ~/Desktop/5g-slicing-embb/controller
python3 zero_touch_controller.py --live
```

---


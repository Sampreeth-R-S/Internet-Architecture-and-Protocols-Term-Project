# 5G URLLC Network Slicing - Complete Command Guide

> Run these commands **in order**. All project source files have already been created.

---

## Phase 1: Install Dependencies & Configure URLLC Slice

### Step 1.1 - Install Node.js 20.x

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

Verify: `node --version`, should show `v20.x.x`

---

### Step 1.2 - Install system dependencies

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

### Step 1.4 - Configure URLLC Slice in Open5GS

You need to edit **two** config files to add the URLLC Slice (`sst: 2`, `sd: 000002`).

#### Edit AMF config:

```bash
sudo nano /etc/open5gs/amf.yaml
```

Find the `plmn_support` section and **change it to:**
```yaml
  plmn_support:
    - plmn_id:
        mcc: 999
        mnc: 70
      s_nssai:
        - sst: 1
          sd: 000001
        - sst: 2
          sd: 000002   # URLLC
```

#### Edit NSSF config:

```bash
sudo nano /etc/open5gs/nssf.yaml
```

Find the `nsi` section and **change it to:**
```yaml
      nsi:
        - uri: http://127.0.0.10:7777
          s_nssai:
            sst: 1
            sd: 000001
        - uri: http://127.0.0.10:7777
          s_nssai:
            sst: 2
            sd: 000002   # URLLC
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

### Step 1.5 - Register 5 URLLC Subscribers via WebUI

Open `http://localhost:9999` (login: **admin** / **1423**) and register **5 subscribers**, one per UE.

For **each subscriber**, click the **"+"** button and fill in:

| Field | Value |
|---|---|
| **Key (K)** | `465B5CE8B199B49FAA5F0A2EE238A6BC` |
| **OPC** | `E8ED289DEBA952E4283B54E88E6183CA` |
| **SST** | `2` |
| **SD** | `000002` |
| **DNN / APN** | `internet` |
| **5QI** | `85` |
| **ARP Priority** | `5` |
| **Session-AMBR DL** | `10` Mbps |
| **Session-AMBR UL** | `5` Mbps |

Use the following **IMSI** values (one per subscriber):

| UE | IMSI | Config File | Assigned IP |
|---|---|---|---|
| UE 1 | `999700000000006` | `ue-urllc1.yaml` | `10.45.0.2` |
| UE 2 | `999700000000007` | `ue-urllc2.yaml` | `10.45.0.3` |
| UE 3 | `999700000000008` | `ue-urllc3.yaml` | `10.45.0.4` |
| UE 4 | `999700000000009` | `ue-urllc4.yaml` | `10.45.0.5` |
| UE 5 | `999700000000010` | `ue-urllc5.yaml` | `10.45.0.6` |

> **Note:** The initial QoS profile (5QI=85, ARP Priority=5) represents the **NORMAL** state. The controller will dynamically switch to the **ELEVATED** profile (5QI=82, ARP Priority=1) when predicted latency exceeds 4 ms, and relax back to NORMAL when stable.

Click **Save** after filling each subscriber form.

---

### Step 1.6 - Build UERANSIM

```bash
cd ~/Desktop/5g-slicing-urllc
git clone https://github.com/aligungr/UERANSIM.git ueransim-src
cd ueransim-src
make
```

Verify: `ls build/nr-gnb build/nr-ue` — both binaries should exist.

---

## Phase 2: Start UERANSIM (gNB + 5 UEs)

You need **7 separate terminals** for this phase.

### Step 2.1 - Terminal 1: Start gNB

```bash
cd ~/Desktop/5g-slicing-urllc/ueransim-src
sudo ./build/nr-gnb -c ../ueransim/gnb-urllc.yaml
```

Expected: `NG Setup procedure is successful`

The gNB connects with:
- **MCC/MNC:** 999/70
- **TAC:** 1
- **Slice:** SST=2, SD=000002 (URLLC)
- **AMF address:** 127.0.0.5:38412

---

### Step 2.2 - Terminals 2–6: Start 5 UEs (one per terminal)

**Terminal 2 — UE 1 (IMSI: 999700000000006):**
```bash
cd ~/Desktop/5g-slicing-urllc/ueransim-src
sudo ./build/nr-ue -c ../ueransim/ue-urllc1.yaml
```

**Terminal 3 — UE 2 (IMSI: 999700000000007):**
```bash
cd ~/Desktop/5g-slicing-urllc/ueransim-src
sudo ./build/nr-ue -c ../ueransim/ue-urllc2.yaml
```

**Terminal 4 — UE 3 (IMSI: 999700000000008):**
```bash
cd ~/Desktop/5g-slicing-urllc/ueransim-src
sudo ./build/nr-ue -c ../ueransim/ue-urllc3.yaml
```

**Terminal 5 — UE 4 (IMSI: 999700000000009):**
```bash
cd ~/Desktop/5g-slicing-urllc/ueransim-src
sudo ./build/nr-ue -c ../ueransim/ue-urllc4.yaml
```

**Terminal 6 — UE 5 (IMSI: 999700000000010):**
```bash
cd ~/Desktop/5g-slicing-urllc/ueransim-src
sudo ./build/nr-ue -c ../ueransim/ue-urllc5.yaml
```

For each UE, expected output: `PDU Session establishment is successful`

Verify tunnel interfaces are created:
```bash
ip addr show | grep uesimtun
# Should show: uesimtun0, uesimtun1, uesimtun2, uesimtun3, uesimtun4
```

---

## Phase 3: Setup Network Namespace & IP Routing

Run `setup_namespace_ip_tables.sh` to configure per-UE routing tables and create the `dn` (Data Network) namespace that the server will run inside.

### Step 3.1 - Run the setup script

```bash
cd ~/Desktop/5g-slicing-urllc
sudo bash setup_namespace_ip_tables.sh
```

What this script does:
1. Assigns `10.45.0.10/16` to the `ogstun` interface
2. Flushes iptables and sets `rp_filter=0` to allow asymmetric routing
3. Creates **5 per-UE routing tables** (`ue0`–`ue4`) mapping each UE IP to its tunnel interface:
   - `10.45.0.2` → `uesimtun0` (table `ue0`)
   - `10.45.0.3` → `uesimtun1` (table `ue1`)
   - `10.45.0.4` → `uesimtun2` (table `ue2`)
   - `10.45.0.5` → `uesimtun3` (table `ue3`)
   - `10.45.0.6` → `uesimtun4` (table `ue4`)
4. Creates the `dn` network namespace, moves `ogstun` into it, and assigns `10.46.0.10/24`
5. Adds return route `10.45.0.0/16` inside the `dn` namespace so the server can reply to all UEs

Verify:
```bash
# Check dn namespace interfaces
sudo ip netns exec dn ip addr show
# Should show ogstun with 10.46.0.10/24

# Check routing rules
ip rule
# Should show 5 per-UE rules with priorities 100–104

# Test end-to-end reachability from UE 2 perspective
ip route get 10.46.0.10 from 10.45.0.3
```

---

## Phase 4: Generate Traffic & Collect Latencies

You need **2 terminals** for this phase.

### Step 4.1 - Terminal A: Start the UDP Server (inside `dn` namespace)

```bash
cd ~/Desktop/5g-slicing-urllc
bash run_server.sh
```

This runs: `sudo ip netns exec dn python3 custom_server.py`

The server:
- Binds to `10.46.0.10:5202` (UDP) inside the `dn` namespace
- Prints per-packet CSV output: `ip,seq,latency_ms,recv,lost`
- Tracks **one-way latency** (sender timestamp embedded in each packet)
- Tracks **packet loss** per UE (via sequence number gaps)

To save the output for the controller to use:
```bash
bash run_server.sh | tee data/training_data.csv
```

---

### Step 4.2 - Terminal B: Start the UDP Client (5 UEs in parallel)

```bash
cd ~/Desktop/5g-slicing-urllc
sudo python3 custom_client.py
```

The client:
- Spawns **5 threads**, one per UE (`uesimtun0`–`uesimtun4`)
- Each UE sends UDP packets to `10.46.0.10:5202` for **3000 seconds**
- Randomly varies **packet size** (64–1024 bytes) and **rate** (10–200 pps)
- Uses `SO_BINDTODEVICE` to pin each thread's socket to its dedicated tunnel interface
- Injects lognormal inter-packet delays (μ=ln(2.5), σ=0.4) to simulate realistic URLLC jitter

Expected output (per UE thread):
```
Interfaces: ['uesimtun0', 'uesimtun1', 'uesimtun2', 'uesimtun3', 'uesimtun4']
[UE1] IP=10.45.0.2, IFACE=uesimtun0
[UE2] IP=10.45.0.3, IFACE=uesimtun1
...
```

---

## Phase 5: Train the LSTM Latency Predictor

```bash
cd ~/Desktop/5g-slicing-urllc
python3 lstm_model.py
```

Expected output:
- Training progress printed every few epochs
- Model saved to `saved/lstm_urllc.pth`

Verify:
```bash
ls saved/
# Should contain: lstm_urllc.pth
```

---

## Phase 6: Run the URLLC Zero-Touch Controller

The controller reads live/historical latency data, uses the trained LSTM to **predict** future latency, and **dynamically adjusts QoS** (5QI + ARP priority) for all URLLC subscribers via the Open5GS WebUI API.

### QoS States

| State | 5QI | ARP Priority | Triggered when |
|---|---|---|---|
| **NORMAL** | 85 | 5 | Predicted latency ≤ 4 ms |
| **ELEVATED (CRITICAL)** | 82 | 1 | Predicted latency > 4 ms |

### Step 6.1 - Install Python dependencies

```bash
pip3 install numpy pandas torch scikit-learn matplotlib requests
# Or use the requirements file:
pip3 install -r lstm_controller_req.txt
```

### Step 6.2 - Run in simulation mode (replays training_data.csv)

```bash
cd ~/Desktop/5g-slicing-urllc/controller
python3 controller.py --mode sim --model ../saved/lstm_urllc.pth --data ../data/training_data.csv
```

Expected output:
```
🔐 Authenticating with Open5GS WebUI...
    [+] CSRF token: ...
    [+] Authentication Successful!

Starting URLLC Simulation | Threshold: 4.0ms
[timestamp] ✅ STABLE: Predicted latency 2.30ms <= 4.0ms.
    ↳ RELAXING Priority -> 5QI: 85, ARP: 5
[timestamp] ⚠️  ALERT: Predicted latency 6.12ms > 4.0ms.
    ↳ ELEVATING Priority -> 5QI: 82, ARP: 1
    [OK] Updated IMSI: 999700000000006
    ...
    [OK] Pushed QoS update to 5 URLLC subscriber(s).
```

### Step 6.3 - Run in live mode (polls data in real time)

```bash
cd ~/Desktop/5g-slicing-urllc/controller
python3 controller.py --mode live --model ../saved/lstm_urllc.pth --data ../data/training_data.csv
```

Expected output every second:
```
[LIVE] Predictions (next 3 steps): [2.45 2.51 2.48]
[LIVE] Warming up... 3 more sample(s) needed.
```

Press `Ctrl+C` to stop. The controller will print total interventions made.

---

## Summary: Full Execution Order

```
Terminal 1  →  sudo ./build/nr-gnb -c ../ueransim/gnb-urllc.yaml
Terminal 2  →  sudo ./build/nr-ue -c ../ueransim/ue-urllc1.yaml
Terminal 3  →  sudo ./build/nr-ue -c ../ueransim/ue-urllc2.yaml
Terminal 4  →  sudo ./build/nr-ue -c ../ueransim/ue-urllc3.yaml
Terminal 5  →  sudo ./build/nr-ue -c ../ueransim/ue-urllc4.yaml
Terminal 6  →  sudo ./build/nr-ue -c ../ueransim/ue-urllc5.yaml
─────────────────────────────────────────────────────────────────
              sudo bash setup_namespace_ip_tables.sh
─────────────────────────────────────────────────────────────────
Terminal A  →  bash run_server.sh | tee data/training_data.csv
Terminal B  →  sudo python3 custom_client.py
─────────────────────────────────────────────────────────────────
              python3 lstm_model.py          (train model)
─────────────────────────────────────────────────────────────────
Terminal C  →  python3 controller/controller.py --mode live
```

---

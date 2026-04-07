# 5G Network Slicing with Dynamic QoS Management

A complete 5G network slicing testbed implementing three distinct slices — **eMBB**, **mMTC**, and **URLLC** — with intelligent zero-touch controllers that compare static, reactive (threshold-based), and LSTM-based proactive resource management.

Built on **Open5GS** (5G core) and **UERANSIM** (RAN simulator), the system trains per-slice LSTM models on live traffic data, then uses predictions to dynamically adjust subscriber QoS parameters (AMBR, 5QI, ARP) via the Open5GS WebUI API.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Step 1: Install System Dependencies](#step-1-install-system-dependencies)
- [Step 2: Configure All Three Slices in Open5GS](#step-2-configure-all-three-slices-in-open5gs)
- [Step 3: Register Subscribers](#step-3-register-subscribers)
- [Step 4: Build UERANSIM](#step-4-build-ueransim)
- [Step 5: Start the gNBs](#step-5-start-the-gnbs)
- [Step 6: Start the UEs](#step-6-start-the-ues)
- [Step 7: Setup Network Namespaces and Routing](#step-7-setup-network-namespaces-and-routing)
- [Step 8: Generate Traffic and Collect Data](#step-8-generate-traffic-and-collect-data)
- [Step 9: Train the LSTM Models](#step-9-train-the-lstm-models)
- [Step 10: Run the Unified Controller (Live Mode)](#step-10-run-the-unified-controller-live-mode)
- [Step 11: Run the Reactive Controller (Baseline)](#step-11-run-the-reactive-controller-baseline)
- [Step 12: Compare Controllers and Visualize Results](#step-12-compare-controllers-and-visualize-results)
- [Project Structure](#project-structure)
- [Slice Comparison Table](#slice-comparison-table)

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                         Open5GS 5G Core                          │
│   AMF ─ SMF ─ UPF ─ NRF ─ NSSF ─ PCF ─ UDM ─ UDR ─ AUSF       │
│                                                                   │
│   Configured Slices:                                              │
│     SST=1  SD=000001  eMBB   (high-bandwidth video streaming)    │
│     SST=2  SD=000002  URLLC  (ultra-reliable low latency)        │
│     SST=3  SD=000003  mMTC   (massive IoT sensor traffic)        │
└────────────────────────────┬──────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────┴────┐        ┌─────┴────┐        ┌────┴────┐
    │  gNB    │        │   gNB    │        │   gNB   │
    │  eMBB   │        │  URLLC   │        │  mMTC   │
    │127.0.0.1│        │127.0.0.3 │        │127.0.0.2│
    └────┬────┘        └────┬─────┘        └────┬────┘
         │                  │                   │
    5 UEs (tun0-4)     5 UEs (tun5-9)     1 UE (tun10)
         │                  │                   │
    ┌────┴────┐        ┌────┴─────┐        ┌────┴────┐
    │Netflix  │        │ UDP      │        │ IoT     │
    │4K Stream│        │ Latency  │        │ Sensor  │
    │Generator│        │ Client   │        │ Traffic │
    └─────────┘        └──────────┘        └─────────┘

    collect_metrics.py   custom_server.py   collect_metrics.py
         │                     │                  │
         ▼                     ▼                  ▼
    embb_traffic_        urllc_timeseries   mmtc_traffic_
    timeseries.csv           .csv           timeseries.csv
         │                     │                  │
         ▼                     ▼                  ▼
    LSTM Predictor       LSTM Predictor      LSTM Predictor
    (throughput)          (latency)          (packet rate)
         │                     │                  │
         └──────────┬──────────┘──────────────────┘
                    ▼
         unified_controller.py  ◄── polls CSVs, predicts, pushes QoS
                    │
                    ▼
            Open5GS WebUI API   ◄── updates AMBR / 5QI / ARP per subscriber
```

---

## Prerequisites

- **OS:** Ubuntu 20.04+ (tested on Ubuntu 22.04)
- **Open5GS** installed and running ([installation guide](https://open5gs.org/open5gs/docs/guide/01-quickstart/))
- **Root/sudo** access (UERANSIM requires `sudo` for tunnel interfaces)
- **Python 3.8+**

---

## Step 1: Install System Dependencies

```bash
# Node.js 20.x (required for Open5GS WebUI)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Build tools and network utilities
sudo apt install -y iperf3 cmake libsctp-dev lksctp-tools git

# Open5GS WebUI
curl -fsSL https://open5gs.org/open5gs/assets/webui/install | sudo -E bash -

# Python ML packages
pip3 install numpy pandas torch scikit-learn matplotlib requests
```

Verify:
```bash
node --version          # v20.x.x
cmake --version
iperf3 --version
```

Open `http://localhost:9999` in a browser — login: **admin** / **1423**

---

## Step 2: Configure All Three Slices in Open5GS

### Edit AMF config

```bash
sudo nano /etc/open5gs/amf.yaml
```

Find the `plmn_support` section and set it to:

```yaml
  plmn_support:
    - plmn_id:
        mcc: 999
        mnc: 70
      s_nssai:
        - sst: 1
          sd: 000001    # eMBB
        - sst: 2
          sd: 000002    # URLLC
        - sst: 3
          sd: 000003    # mMTC
```

### Edit NSSF config

```bash
sudo nano /etc/open5gs/nssf.yaml
```

Set the `nsi` section to:

```yaml
      nsi:
        - uri: http://127.0.0.10:7777
          s_nssai:
            sst: 1
            sd: 000001    # eMBB
        - uri: http://127.0.0.10:7777
          s_nssai:
            sst: 2
            sd: 000002    # URLLC
        - uri: http://127.0.0.10:7777
          s_nssai:
            sst: 3
            sd: 000003    # mMTC
```

### Restart services

```bash
sudo systemctl restart open5gs-amfd open5gs-nssfd
sudo systemctl status open5gs-amfd open5gs-nssfd
```

---

## Step 3: Register Subscribers

Open `http://localhost:9999` (login: **admin** / **1423**) and register subscribers using the **"+"** button.

All subscribers share the same crypto material:
- **Key (K):** `465B5CE8B199B49FAA5F0A2EE238A6BC`
- **OPC:** `E8ED289DEBA952E4283B54E88E6183CA`

### eMBB Subscribers (5 UEs)

Register 5 subscribers with these IMSIs: `999700000000001` through `999700000000005`

| Field | Value |
|---|---|
| SST | `1` |
| SD | `000001` |
| DNN | `internet` |
| Session-AMBR DL | `100` Mbps |
| Session-AMBR UL | `50` Mbps |

### URLLC Subscribers (5 UEs)

Register 5 subscribers with these IMSIs:

| UE | IMSI | Config File |
|---|---|---|
| UE 1 | `999700000000006` | `ue-urllc1.yaml` |
| UE 2 | `999700000000007` | `ue-urllc2.yaml` |
| UE 3 | `999700000000008` | `ue-urllc3.yaml` |
| UE 4 | `999700000000009` | `ue-urllc4.yaml` |
| UE 5 | `999700000000010` | `ue-urllc5.yaml` |

| Field | Value |
|---|---|
| SST | `2` |
| SD | `000002` |
| DNN | `internet` |
| 5QI | `85` |
| ARP Priority | `5` |
| Session-AMBR DL | `10` Mbps |
| Session-AMBR UL | `5` Mbps |

> The initial QoS profile (5QI=85, ARP=5) is the **NORMAL** state. The controller dynamically elevates to **CRITICAL** (5QI=82, ARP=1) when predicted latency exceeds the threshold.

### mMTC Subscriber (1 UE)

Register 1 subscriber with IMSI: `999700000000011`

| Field | Value |
|---|---|
| SST | `3` |
| SD | `000003` |
| DNN | `internet` |
| Session-AMBR DL | `10` Mbps |
| Session-AMBR UL | `5` Mbps |

---

## Step 4: Build UERANSIM

```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-embb
git clone https://github.com/aligungr/UERANSIM.git ueransim-src
cd ueransim-src
make
```

Verify: `ls build/nr-gnb build/nr-ue` — both binaries should exist.

Symlink the build for the other slices so they share the same binary:

```bash
ln -s ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-embb/ueransim-src \
      ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-mmtc/ueransim-src

ln -s ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-embb/ueransim-src \
      ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-urllc/ueransim-src
```

---

## Step 5: Start the gNBs

A convenience script launches all three gNBs in the background:

```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project
sudo bash start_gnbs.sh
```

This starts:
- **eMBB gNB** — `gnb-embb.yaml` (127.0.0.1, SST=1, SD=000001)
- **mMTC gNB** — `gnb-mmtc.yaml` (127.0.0.2, SST=3, SD=000003)
- **URLLC gNB** — `gnb-urllc.yaml` (127.0.0.3, SST=2, SD=000002)

Expected: `NG Setup procedure is successful` for each gNB.

Press `Ctrl+C` to stop all gNBs.

---

## Step 6: Start the UEs

In a **separate terminal**, launch all 11 UEs:

```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project
sudo bash start_ues.sh
```

This launches:
- **5 eMBB UEs** → tunnel interfaces `uesimtun0`–`uesimtun4`
- **5 URLLC UEs** → tunnel interfaces `uesimtun5`–`uesimtun9`
- **1 mMTC UE** → tunnel interface `uesimtun10`

Expected: `PDU Session establishment is successful` for each UE.

Verify:
```bash
ip addr show | grep uesimtun
# Should show uesimtun0 through uesimtun10
```

---

## Step 7: Setup Network Namespaces and Routing

This configures per-UE routing tables and creates the `dn` (Data Network) namespace needed by the URLLC server:

```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project
sudo bash setup_namespace_ip_tables.sh
```

What this does:
1. Assigns `10.45.0.10/16` to `ogstun`
2. Flushes iptables and disables `rp_filter` for asymmetric routing
3. Creates **11 per-UE routing tables** (`ue0`–`ue10`), each mapping a UE IP to its tunnel interface
4. Creates the `dn` network namespace, moves `ogstun` into it, assigns `10.46.0.10/24`
5. Adds return route `10.45.0.0/16` inside the `dn` namespace

Verify:
```bash
sudo ip netns exec dn ip addr show        # ogstun with 10.46.0.10/24
ip rule                                     # per-UE rules (priorities 100-110)
ip route get 10.46.0.10 from 10.45.0.3    # routable through correct tunnel
```

---

## Step 8: Generate Traffic and Collect Data

You need **5 terminals** (or use background processes) to generate traffic for all 3 slices simultaneously.

### eMBB Traffic (2 terminals)

**Terminal A — Metrics collector:**
```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-embb/traffic_gen
python3 collect_metrics.py
```

**Terminal B — Netflix 4K streaming simulator:**
```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-embb/traffic_gen
python3 netflix_4k_streaming.py
```

Output: `5g-slicing-embb/data/embb_traffic_timeseries.csv` (column: `throughput_mbps`)

### URLLC Traffic (2 terminals)

**Terminal C — UDP latency server** (runs inside `dn` namespace):
```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-urllc
bash run_server.sh | tee data/training_data.csv
```

**Terminal D — UDP latency client** (5 parallel UE threads):
```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-urllc
sudo python3 custom_client.py
```

Output: `5g-slicing-urllc/data/training_data.csv` (column: `latency_ms`)

### mMTC Traffic (2 terminals)

**Terminal E — Metrics collector:**
```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-mmtc/traffic_gen
python3 collect_metrics.py
```

**Terminal F — IoT sensor simulator:**
```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-mmtc/traffic_gen
python3 mmtc_sensor_traffic.py
```

Output: `5g-slicing-mmtc/data/mmtc_traffic_timeseries.csv` (column: `packet_rate`)

---

## Step 9: Train the LSTM Models

Train each slice's LSTM predictor on the collected data:

```bash
# eMBB — predicts throughput (Mbps), 24-step window, 6-step horizon
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-embb/models
python3 lstm_predictor.py

# mMTC — predicts packet rate (pps), 60-step window, 6-step horizon
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-mmtc/models
python3 lstm_predictor.py

# URLLC — predicts latency (ms), 3 features (raw + rolling mean + rolling std)
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-urllc
python3 models/lstm_predictor.py
```

Verify models are saved:
```bash
ls 5g-slicing-embb/models/saved/lstm_embb.pth
ls 5g-slicing-mmtc/models/saved/lstm_mmtc.pth
ls 5g-slicing-urllc/saved/lstm_urllc.pth
```

---

## Step 10: Run the Unified Controller (Live Mode)

The unified controller manages all 3 slices concurrently using multiprocessing. Each slice runs in its own process, polling its CSV, predicting with its LSTM model, and pushing QoS updates to Open5GS.

```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project
chmod +x run_unified_controller.sh
./run_unified_controller.sh
```

Or run directly:
```bash
python3 unified_controller.py \
    --mode live \
    --urllc-data 5g-slicing-urllc/data/urllc_timeseries.csv \
    --mmtc-data  5g-slicing-mmtc/data/mmtc_traffic_timeseries.csv \
    --embb-data  5g-slicing-embb/data/embb_traffic_timeseries.csv \
    --urllc-model 5g-slicing-urllc/saved/lstm_urllc.pth \
    --mmtc-model  5g-slicing-mmtc/models/saved/lstm_mmtc.pth \
    --embb-model  5g-slicing-embb/models/saved/lstm_embb.pth
```

**Per-slice actions:**

| Slice | Predicts | Actions | Range |
|---|---|---|---|
| eMBB | Throughput (Mbps) | expand/contract bandwidth | 100–2500 Mbps DL |
| mMTC | Packet rate (pps) | expand/contract capacity | 200–2000 pps |
| URLLC | Latency (ms) | elevate/relax 5QI & ARP | 5QI 85↔82, ARP 5↔1 |

Decision logs are saved to:
- `unified_decisions_embb.json`
- `unified_decisions_mmtc.json`
- `unified_decisions_urllc.json`

Press `Ctrl+C` to stop all three slice controllers.

---

## Step 11: Run the Reactive Controller (Baseline)

The reactive controller uses the same architecture but makes decisions based on **current metric values** (no LSTM prediction):

```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project
python3 reactive_controller.py \
    --mode live \
    --urllc-data 5g-slicing-urllc/data/urllc_timeseries.csv \
    --mmtc-data  5g-slicing-mmtc/data/mmtc_traffic_timeseries.csv \
    --embb-data  5g-slicing-embb/data/embb_traffic_timeseries.csv
```

Decision logs are saved to:
- `reactive_decisions_embb.json`
- `reactive_decisions_mmtc.json`
- `reactive_decisions_urllc.json`

---

## Step 12: Compare Controllers and Visualize Results

### Generate comparison metrics and plots

```bash
cd ~/Desktop/Internet-Architecture-and-Protocols-Term-Project
python3 compare_controllers.py
```

This produces **17 visualizations** in `comparison_visualizations/`:

| Category | Plots |
|---|---|
| **eMBB** | Throughput timeseries, allocation vs demand, bar summary, decision/BW timeline |
| **mMTC** | Packet rate timeseries, capacity timeline, bar summary, decision/capacity timeline |
| **URLLC** | Latency CDF, latency boxplot, SLA violations, per-UE latency, decision/latency timeline |
| **Cross-Slice** | Radar chart, decision distribution, decision volume breakdown, summary table |

### Generate per-controller plots

```bash
# Unified (LSTM) controller plots
python3 unified_controller_plot.py
# → Output in unified_visualizations/

# Per-slice dashboards
cd 5g-slicing-embb/visualization && python3 dashboard.py && cd ../..
cd 5g-slicing-mmtc/visualization && python3 dashboard.py && cd ../..
```

### Run per-slice baseline comparisons

```bash
cd 5g-slicing-embb/evaluation && python3 compare_baselines.py && cd ../..
cd 5g-slicing-mmtc/evaluation && python3 compare_baselines.py && cd ../..
```

---

## Project Structure

```
├── unified_controller.py          # LSTM-based proactive controller (all 3 slices)
├── reactive_controller.py         # Threshold-based reactive controller (baseline)
├── compare_controllers.py         # Performance comparison & visualization generator
├── unified_controller_plot.py     # Post-run visualization for unified controller
├── run_unified_controller.sh      # Convenience launcher for unified controller
├── start_gnbs.sh                  # Launches all 3 gNBs in background
├── start_ues.sh                   # Launches all 11 UEs (5 eMBB + 5 URLLC + 1 mMTC)
├── setup_namespace_ip_tables.sh   # Network namespace and per-UE routing setup
│
├── 5g-slicing-embb/               # eMBB slice (SST=1, SD=000001)
│   ├── controller/                #   Zero-touch controller
│   ├── models/                    #   LSTM predictor + saved weights
│   ├── traffic_gen/               #   collect_metrics.py, netflix_4k_streaming.py
│   ├── evaluation/                #   Static vs Reactive vs Proactive comparison
│   ├── visualization/             #   Dashboard plots
│   ├── ueransim/                  #   gNB + 5 UE YAML configs
│   ├── ueransim-src/              #   UERANSIM build (shared via symlinks)
│   └── data/                      #   Timeseries CSVs, iperf logs
│
├── 5g-slicing-urllc/              # URLLC slice (SST=2, SD=000002)
│   ├── controller/                #   URLLC-specific QoS controller
│   ├── models/                    #   3-feature LSTM (latency + rolling stats)
│   ├── custom_client.py           #   UDP latency test client (5 UE threads)
│   ├── custom_server.py           #   UDP server (runs in dn namespace)
│   ├── run_server.sh              #   Launches server in dn namespace
│   ├── ueransim/                  #   gNB + 5 UE YAML configs
│   ├── saved/                     #   Trained model weights
│   └── data/                      #   Latency timeseries CSV
│
├── 5g-slicing-mmtc/               # mMTC slice (SST=3, SD=000003)
│   ├── controller/                #   Packet-rate based capacity controller
│   ├── models/                    #   60-step window LSTM
│   ├── traffic_gen/               #   collect_metrics.py, mmtc_sensor_traffic.py
│   ├── evaluation/                #   Baseline comparison
│   ├── visualization/             #   Dashboard plots
│   ├── ueransim/                  #   gNB + 1 UE YAML config
│   └── data/                      #   Packet rate timeseries CSV
│
├── comparison_visualizations/     # 17 cross-controller comparison plots
├── unified_visualizations/        # Per-slice unified controller plots
├── reactive_visualizations/       # Per-slice reactive controller plots
│
├── unified_decisions_*.json       # LSTM controller decision logs
├── reactive_decisions_*.json      # Reactive controller decision logs
└── controller_comparisons.txt     # Console comparison output
```

---

## Slice Comparison Table

| Aspect | eMBB | URLLC | mMTC |
|---|---|---|---|
| **SST / SD** | 1 / 000001 | 2 / 000002 | 3 / 000003 |
| **Purpose** | 4K video streaming | Ultra-low latency control | Massive IoT sensors |
| **UEs** | 5 | 5 | 1 (represents 1000+ devices) |
| **Tunnel Interfaces** | uesimtun0–4 | uesimtun5–9 | uesimtun10 |
| **gNB Loopback** | 127.0.0.1 | 127.0.0.3 | 127.0.0.2 |
| **Session-AMBR DL/UL** | 100/50 Mbps | 10/5 Mbps | 10/5 Mbps |
| **LSTM Predicts** | Throughput (Mbps) | Latency (ms) | Packet rate (pps) |
| **Controller Action** | Expand/contract BW | Elevate/relax 5QI+ARP | Expand/contract capacity |
| **5QI** | 8/9 (GBR) | 85→82 (dynamic) | 79 (non-GBR) |
| **Traffic Generator** | netflix_4k_streaming.py | custom_client.py | mmtc_sensor_traffic.py |
| **Metrics Collector** | collect_metrics.py | custom_server.py | collect_metrics.py |

---

## Quick Start (Full Pipeline Summary)

```bash
# 1. Start all gNBs
sudo bash start_gnbs.sh                    # Terminal 1

# 2. Start all UEs
sudo bash start_ues.sh                     # Terminal 2

# 3. Setup routing
sudo bash setup_namespace_ip_tables.sh     # any terminal

# 4. Start traffic generators
# eMBB
cd 5g-slicing-embb/traffic_gen
python3 collect_metrics.py &               # Terminal 3
python3 netflix_4k_streaming.py &          # Terminal 4

# URLLC
cd ../../5g-slicing-urllc
bash run_server.sh | tee data/training_data.csv &   # Terminal 5
sudo python3 custom_client.py &                      # Terminal 6

# mMTC
cd ../5g-slicing-mmtc/traffic_gen
python3 collect_metrics.py &               # Terminal 7
python3 mmtc_sensor_traffic.py &           # Terminal 8

# 5. (After collecting enough data) Train models
cd ../../5g-slicing-embb/models  && python3 lstm_predictor.py
cd ../../5g-slicing-mmtc/models  && python3 lstm_predictor.py
cd ../../5g-slicing-urllc        && python3 models/lstm_predictor.py

# 6. Run the unified controller
cd ..
./run_unified_controller.sh                # Terminal 9

# 7. Compare and visualize
python3 compare_controllers.py
```

# Bistatic Localization System

This directory contains scripts for running bistatic UWB localization experiments with multiple nodes.

## System Overview

The bistatic localization system consists of:
- **Node 0 (SPEAKER)**: Transmitter that sends UWB pulses
- **Nodes 1, 2, 3 (LISTENER)**: Receivers that capture Channel Impulse Response (CIR) data

## Files

### Control Script
- **`cir_control_remote_nodes_localizing.py`**: Main controller that orchestrates all remote nodes
  - Starts listener nodes first (1, 2, 3) with LCFG and LISTENER commands
  - Then starts speaker node (0) with SCFG and SPEAKER commands
  - Waits for duration to complete
  - Retrieves CIR files from all listener nodes
  - Saves files to timestamped subdirectory with node suffixes
  - Copies configuration file to output directory
  - Power cycles devices after completion

### Remote Runner Scripts
- **`remote_runner_listener_localizing.py`**: Runs on listener nodes (1, 2, 3)
  - Sends LCFG configuration command
  - Sends LISTENER command to start receiving
  - Collects and saves CIR data with diagnostics
  
- **`remote_runner_speaker_localizing.py`**: Runs on speaker node (0)
  - Sends SCFG configuration command
  - Sends SPEAKER command to start transmitting
  - Runs for specified duration

### Configuration
- **`../config/cir_localizing.yaml`**: Configuration file for localization experiments
  - Radio parameters (channel, CIR samples, FPS, duration, etc.)
  - Room/environment parameters (layout, humans, activity, tag position)

## Usage

### Basic Usage
```bash
python cir_control_remote_nodes_localizing.py --suffix experiment1
```

### With Custom Configuration
```bash
python cir_control_remote_nodes_localizing.py \
    --config cir_localizing.yaml \
    --suffix myexperiment \
    --time 30 \
    --fps 1000 \
    --channel 5
```

### Command Line Arguments
- `--config`: Path to configuration YAML file (default: cir_localizing.yaml)
- `--save`: Enable saving (always enabled for listeners)
- `--en`: Enable tag on remote devices
- `--suffix`: Suffix for experiment identification
- `--time`: Override duration in seconds
- `--fps`: Override frames per second
- `--channel`: Override UWB channel
- `--cir_sample`: Override CIR sample size
- `--cir_start_index`: Override CIR start index
- `--delayTX`: Override TX delay offset
- `--xtal_trim`: Override crystal trim value
- `--boards`: Specify board IDs (default: 0, 1, 2, 3)

## Output Structure

Each experiment creates a timestamped directory in `cir_files/`:
```
cir_files/
├── 20251029_143522_experiment1/
│   ├── cir_localizing.yaml              # Copy of config file
│   ├── diag/
│   │   ├── diag_data_rx_node1.npy       # Diagnostics from node 1
│   │   ├── diag_data_rx_node2.npy       # Diagnostics from node 2
│   │   └── diag_data_rx_node3.npy       # Diagnostics from node 3
│   └── rx/
│       ├── cir_data_rx_node1.npy        # CIR data from node 1
│       ├── cir_data_rx_node2.npy        # CIR data from node 2
│       └── cir_data_rx_node3.npy        # CIR data from node 3
└── experiment_summary_localizing.csv     # Summary of all experiments
```

## Board Configuration

Edit `BOARD_CONFIG` in `cir_control_remote_nodes_localizing.py` to match your setup:

```python
BOARD_CONFIG = {
    0: {"ip": "192.168.0.115", "role": "SPEAKER", ...},  # Transmitter
    1: {"ip": "192.168.0.115", "role": "LISTENER", ...}, # Receiver 1
    2: {"ip": "192.168.0.131", "role": "LISTENER", ...}, # Receiver 2
    3: {"ip": "192.168.0.120", "role": "LISTENER", ...}, # Receiver 3
}
```

## Execution Flow

1. **Initialization**: Load configuration and parse parameters
2. **Start Listeners**: Connect to nodes 1, 2, 3 and send LISTENER commands
3. **Wait**: 2-second delay for listener initialization
4. **Start Speaker**: Connect to node 0 and send SPEAKER command
5. **Capture**: Wait for specified duration while collecting data
6. **Retrieve**: Download CIR files from all listener nodes
7. **Power Cycle**: Reset all devices via USB hub control
8. **Save**: Store all files in timestamped directory with config copy
9. **Log**: Update CSV summary with experiment metadata

## Key Differences from TDMA Sensing

- **Sequential Start**: Listeners start before speaker (vs simultaneous in TDMA)
- **Role-Based**: Explicit SPEAKER/LISTENER roles (vs all nodes equal in TDMA)
- **File Naming**: Uses `_nodeN` suffix (vs `_NodeN` in TDMA)
- **Config File**: `cir_localizing.yaml` (vs `cir_sensing.yaml`)
- **CSV Summary**: `experiment_summary_localizing.csv` (vs `experiment_summary.csv`)
- **No Speaker Files**: Only listener nodes save CIR data

## Remote Runner Deployment

The remote runner scripts need to be deployed to the Raspberry Pi devices:

### For Listener Nodes (1, 2, 3)
```bash
# Copy to /home/icon/uwb/ on each listener Pi
scp remote_runner_listener_localizing.py icon@192.168.0.XXX:/home/icon/uwb/
```

### For Speaker Node (0)
```bash
# Copy to /home/icon/uwb/UWB_code/ on speaker Pi
scp remote_runner_speaker_localizing.py icon@192.168.0.115:/home/icon/uwb/UWB_code/
```

## Troubleshooting

- **Connection Issues**: Check IP addresses in `BOARD_CONFIG`
- **File Not Found**: Ensure remote runner scripts are deployed correctly
- **USB Power Cycle Fails**: Verify `uhubctl` is installed on Raspberry Pi
- **Missing Data**: Check that listener nodes are receiving transmissions from speaker

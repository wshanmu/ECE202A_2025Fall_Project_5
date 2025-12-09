# DeskSense

## RTC Configure

Use an Arduino UNO, first connects I2C with the evaluation board, as well as the 3V3 and GND, then runs `Configure_RTC.ino`. After configuration, connects `CLKOUT` to UNO PIN 2, and removes the I2C connection, runs `Verify_RTC.ino` to verify the frequency.

In the PCB version, one might need to manually disable the voltage from SCL and SDA to make it work normally.

## Bi-static Implementation

Two seperate Raspberry Pi controling a DWM3001CDK, one acts as the TX and one acts as the RX.
The remote RPI configuration:

```python
# === Remote Raspberry Pi Configuration ===
REMOTE_IP_TX = "192.168.0.115"
USERNAME = "icon"
PASSWORD = "icon"
REMOTE_SERIAL_PORT_TX = "/dev/ttyACM0"
REMOTE_SCRIPT_PATH_TX = "UWB_code/remote_runner_speaker.py"

REMOTE_IP_RX = "192.168.0.120"
USERNAME = "icon"
PASSWORD = "icon"
REMOTE_SERIAL_PORT_RX = "/dev/ttyACM0"
REMOTE_SCRIPT_PATH_RX = "remote_runner_listener.py"
```

The UWB Evaluation boards are flashed with a heavily modified firmware, which supports two main mode: `SPEAKER` and `LISTENER`:

- `SPEAKER` mode: the board will continously send UWB packets. Parameters to be specified: FPS {100, 1000}, CHAN {5, 9}, DURATION, DELAYTX {2,4,6,8,etc.} 
- `LISTENER` mode: the board will start receiving all frames, and report the following diagnostics information:

```python
    diag = {
    'index_fp_u32': unpacked[0], # first path index estimation, uint32_t 
    'accumCount': unpacked[1], # how many preamble being accumulated, uinit16_t
    'plos': unpacked[2], # LoS phase estimation, int16_t
    'D': unpacked[3], # frame index, uint8_t
    'timestamp': None, # 5 bytes uint8_t
    }
```

To communicate with the board directly for debug, using: `tio /dev/ttyACM0 -b 1000000`

To use Jlink for debug on Ubuntu: 
run `JLinkExe -if SWD` and `connect` first, then on a new terminal, run `JLinkRTTClient`

To start data collection, `cir_control_remote.py`:

```cmd
options:
  --save                Save the data to file
  --en                  whether the tag is enable or not
  --suffix SUFFIX       Suffix for filename
  --time TIME           Duration time in seconds
  --fps FPS             Frames per second for remote speaker
  --delayTX DELAYTX     Offset to Delay TX transmission
  --channel CHANNEL     Channel number for experiment {5, 9}
  --cir_sample CIR_SAMPLE CIR sample size (The maximum point to make sure no frame lost is 29)
  --cir_start_index CIR_START_INDEX CIR start index # TODO: something wrong, need to manually set in the firmware
```

The script will connect to both RPI and initialize the transmission. When reach duration time, it automatically get the saved CIR and diag files from the remote RPI.

Example:

```cmd
python ./bistatic_collection/cir_control_remotes.py --fps 1024 --time 2 --delayTX 6 --save --channel 5 --cir_sample 30 --suffix hi
```

```cmd
/home/shanmu/miniconda3/envs/py10/bin/python /home/shanmu/Projects/DeskSense/bistatic_collection/cir_control_remote_nodes_localizing.py --boards 0 1 --channel 9 --time 150 --suffix "cir24_exp0" --fps 1024 --cir_sample 24 --delayTX 6 --xtal_trim 25
```

The `subprocess_data_collection.py` is a wrapper to run `cir_control_remote.py` multiple times. 


## Bi-static TDMA Scheme Sensing Data Collection

Run `./venv/bin/python ./remote_runner_node.py --mode TDMA --save --cir_sample 24` on the Raspberry Pi saperately to collect the data

or

```cmd
python tdma_sensing/cir_control_remote_nodes_sensing.py --boards 0 1 2 3 --time 60 --suffix static
```

## Radar Implementation
The radar implementation relies on [QM35-SDK](https://gitlab.com/qorvo_sdk/public/devkits/qm35-sdk). Refer the way to configure the radar and set up data collection.

## Tag Detection:
Run `tag_detection.ipynb` to detect the tag with a collected bi-static CIR data, and `tag_detection.ipynb` with the radar one.

Specify parameters in `config/cir_processing.yaml`

## CIR Data Processing

### Configuration

All processing parameters are centralized in `config/cir_sensing_processing.yaml`. You only need to specify the `input_folder`, and the script will automatically derive file paths.

**Key parameters:**

- `input_folder`: Path to the data folder (e.g., `"./tdma_sensing/cir_files/20251103_135727_loc2_findTap"`)
- `upsample_factor`: FFT upsampling factor (default: 32)
- `discard_reading`: Number of initial frames to discard (default: 100)
- `zero_padded_num`: Zero padding amount (default: 0)
- `savgol_window_length`: Savitzky-Golay filter window for alignment (default: 51)

### Single Folder Processing

Process a single folder by updating the config file and running:

```bash
python process_sensing_data.py
```

Or override the config from command line without editing the YAML:

```bash
python process_sensing_data.py input_folder="./tdma_sensing/cir_files/20251103_135727_loc2_findTap"
```

### Batch Processing Multiple Folders

Use the Python batch processing script to automatically process multiple folders:

**Process all folders from a specific date:**

```bash
python batch_process_folders.py --pattern "20251103_*"
```

**Process only occupied folders:**

```bash
python batch_process_folders.py --pattern "*occupied*"
```

**Process specific folders:**

```bash
python batch_process_folders.py --folders \
    "./tdma_sensing/cir_files/20251103_135727_loc2_findTap" \
    "./tdma_sensing/cir_files/20251103_140957_loc3_occupied_3"
```

**Exclude certain patterns:**

```bash
python batch_process_folders.py --pattern "20251103_*" --exclude "findTap" "test"
```

**Override config parameters for batch processing:**

```bash
python process_sensing_data.py \
    input_folder="./tdma_sensing/cir_files/20251103_135727_loc2_findTap" \
    upsample_factor=64 \
    savgol_window_length=101
```

### Output

The processing pipeline will:

- Save processed CIR data to: `./tdma_sensing/cir_files/processed_cir/{folder_name}_cir_phased.npy`
    - the CIR data would have the shape of `(NUM_TX, NUM_RX, T, N)`, where N is the tap index (upsampled). The dtype is `complex128`
- Save 6 waterfall plots (one for each TX-RX pair) to the input folder
- Display progress with 7 processing steps and a summary

### Processing Pipeline

1. Load CIR data from multiple nodes
2. Compute fractional shifts from diagnostic data
3. Upsample CIR by specified factor using FFT
4. Apply fractional shifts and normalize
5. Reorganize by TX-RX pairs
6. Align CIR using zero-crossing method
7. Apply phase correction

## Model Training and Hydra Configuration:

After processing sensing CIR files, update the yaml file in `src/data` to specify the source files and corresponding labels for them. 

Run `train_multiDesk.py` to train and test a model with default parameters. The parameters are specified in `config` folder, with Hydra based yaml files.
import serial
import time
import argparse
import numpy as np
import struct
from tqdm import tqdm
import os

# Remote runner for listener nodes in bistatic localization setup
# Sends LCFG and LISTENER commands to UWB device
# Parses and saves CIR data from the device

def send_command(channel, cir_sample=28, cir_start_index=0):
    if MODE not in ['LISTENER']:
        raise ValueError("Invalid mode. Use 'LISTENER'.")
    
    CFG_command = f"LCFG -CHAN={channel} -PCODE=10 -XTALTRIM=22 -CIRSAMPLE={cir_sample} -CIRSI={cir_start_index}\n"
    ser.write(CFG_command.encode())
    time.sleep(0.1)
    
    command = f"LISTENER\n"
    ser.write(command.encode())
    print(f"Sent 'LISTENER' command to the device with channel={channel}, cir_sample={cir_sample}, cir_start_index={cir_start_index}.")
    time.sleep(0.5)

def unpack_24bit_signed(data_bytes):
    value = int.from_bytes(data_bytes, byteorder='little', signed=False)
    if value & 0x800000:
        value -= 0x1000000
    return value

def to_int24_signed(u):
    s = u.astype(np.int32)
    neg_mask = (s & 0x800000) != 0
    s[neg_mask] -= 0x1000000
    return s

def unpack_24bit_signed_vectorized(payload: bytes) -> np.ndarray:
    """
    Parse interleaved 24-bit signed real/imag samples into a complex64 numpy array.

    payload layout per sample (little-endian):
        [r0, r1, r2, i0, i1, i2]  (each r*/i* is uint8)
    """

    # Total samples
    num_samples = len(payload) // BYTES_PER_SAMPLE
    if len(payload) % BYTES_PER_SAMPLE != 0:
        raise ValueError("Payload length is not a multiple of 6 bytes per sample.")

    # 1) Interpret raw bytes as uint8 and reshape
    data_u8 = np.frombuffer(payload, dtype=np.uint8)
    data_u8 = data_u8.reshape(num_samples, BYTES_PER_SAMPLE)  # shape: (N, 6)

    # 2) Split real and imag parts (still uint8)
    real_u8 = data_u8[:, 0:3]  # (N, 3)
    imag_u8 = data_u8[:, 3:6]  # (N, 3)

    # 3) Combine 3 bytes into 24-bit unsigned integers (little-endian)
    # value = b0 + (b1 << 8) + (b2 << 16)
    real_u32 = (real_u8[:, 0].astype(np.uint32)
                | (real_u8[:, 1].astype(np.uint32) << 8)
                | (real_u8[:, 2].astype(np.uint32) << 16))

    imag_u32 = (imag_u8[:, 0].astype(np.uint32)
                | (imag_u8[:, 1].astype(np.uint32) << 8)
                | (imag_u8[:, 2].astype(np.uint32) << 16))

    # 4) Convert 24-bit unsigned to signed (2's complement)
    # if value & 0x800000: value -= 0x1000000
    real_i32 = to_int24_signed(real_u32)
    imag_i32 = to_int24_signed(imag_u32)

    # 5) Convert to complex64 (float32 real & imag)
    real_f32 = real_i32.astype(np.float32)
    imag_f32 = imag_i32.astype(np.float32)

    cir_complex = real_f32 + 1j * imag_f32
    return cir_complex.astype(np.complex64)


def parse_diag_packet(payload):
    if len(payload) != DIAG_PACKET_SIZE:
        print("Invalid packet length")
        ser.write(b"STOP\n")
        return None
    try:
        unpacked = struct.unpack('<IHh6B', payload)  # I: uint32_t, H: uint16_t, h: int16_t, B: uint8_t
        diag = {
            'index_fp_u32': unpacked[0],
            'accumCount': unpacked[1],
            'plos': unpacked[2],
            'D': unpacked[3],
            'timestamp': None,
            'localhost_timestamp': time.time()
        }
        ts_bytes = bytes(unpacked[-5:])
        raw_timestamp = int.from_bytes(ts_bytes, byteorder='little')
        unit = 1 / (128 * 499.2e6)
        diag['timestamp'] = raw_timestamp * unit * 1000
        return diag
    except struct.error as e:
        print(f"Struct unpacking error: {e}")
        return None


parser = argparse.ArgumentParser(description='CIR Extraction Script for Listener (Localizing)')
parser.add_argument('--port', type=str, default='/dev/ttyACM0', help='Serial port to use')
parser.add_argument('--baud', type=int, default=1000000, help='Baud rate for serial communication')
parser.add_argument('--mode', type=str, default='LISTENER', help='Mode to set for the device')
parser.add_argument('--save', action='store_true', help='Save the data to file')
parser.add_argument('--en', action='store_true', help='Whether the tag is enabled or not')
parser.add_argument('--suffix', type=str, default='', help='Suffix for filename')
parser.add_argument('--time', type=int, default=10, help='Duration time in seconds')
parser.add_argument('--cir_sample', type=int, default=24, help='CIR sample size')
parser.add_argument('--cir_start_index', type=int, default=0, help='CIR start index')
parser.add_argument('--channel', type=int, default=9, help='Channel number')
parser.add_argument('--fps', type=int, default=1024, help='Frames per second')

args = parser.parse_args()
PORT = args.port
BAUD = args.baud
MODE = args.mode
SAVE = args.save
NUM_SAMPLES = args.cir_sample
CIR_START_INDEX = args.cir_start_index
CHANNEL = args.channel
EN = args.en
SUFFIX = args.suffix
SYNC_HEADER = b'C'
BYTES_PER_SAMPLE = 6
PACKET_SIZE = NUM_SAMPLES * BYTES_PER_SAMPLE
DIAG_PACKET_SIZE = 9 + 5
SAVING_PATH = './cir_files/'
os.makedirs(SAVING_PATH + 'rx', exist_ok=True)
os.makedirs(SAVING_PATH + 'diag', exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=1)

duration_time = args.time
frame_rate = args.fps
stopping_points = int(duration_time * frame_rate)

# RX Storage
cir_storage_rx = np.zeros((stopping_points, NUM_SAMPLES), dtype=np.complex64)
accumCount_storage_rx = np.zeros((stopping_points,), dtype=np.uint32)
index_fp_u32_storage_rx = np.zeros((stopping_points,), dtype=np.uint32)
D_storage_rx = np.zeros((stopping_points,), dtype=np.uint32)
p_los_storage_rx = np.zeros((stopping_points,), dtype=np.int16)
timestamp_storage_rx = np.zeros((stopping_points,), dtype=np.float64)
local_host_timestamp_storage_rx = np.zeros((stopping_points,), dtype=np.float64)

points_rx = 0

def graceful_exit():
    try:
        if ser.is_open:
            print("Sending STOP command before exiting...")
            ser.write(b"STOP\n")
            time.sleep(0.2)
            ser.close()
            print("Serial port closed.")
        if SAVE:
            print("Saving data...")
            suffix = '_EN' + SUFFIX if EN else SUFFIX

            # Save RX data
            np.save(f"{SAVING_PATH}/rx/cir_data_rx{suffix}.npy", cir_storage_rx[:points_rx])
            diag_storage_rx = [{
                'index_fp_u32': index_fp_u32_storage_rx[i],
                'accumCount': accumCount_storage_rx[i],
                'D': D_storage_rx[i],
                'timestamp': timestamp_storage_rx[i],
                'p_los': p_los_storage_rx[i],
            } for i in range(points_rx)]
            np.save(f"{SAVING_PATH}/diag/diag_data_rx{suffix}.npy", diag_storage_rx)
            print(f"Saved RX data with {points_rx} points.")
        
    except Exception as e:
        print(f"Error during exit: {e}")


if __name__ == "__main__":
    send_command(CHANNEL, NUM_SAMPLES, CIR_START_INDEX)
    
    rx_buffer = bytearray()
    start_time = time.time()
    end_time = start_time + args.time

    try:
        # Initialize progress bar
        progress_bar = tqdm(total=stopping_points, desc="Processing", unit="frames")

        while time.time() < end_time:
            # Read available data from RX
            rx_data = ser.read(ser.in_waiting)
            rx_buffer += rx_data

            # Process RX buffer
            while True:
                sync_pos = rx_buffer.find(SYNC_HEADER)
                if sync_pos == -1:
                    break
                if sync_pos > 0:
                    rx_buffer = rx_buffer[sync_pos:]
                if len(rx_buffer) < 1 + PACKET_SIZE + DIAG_PACKET_SIZE:
                    break
                payload = rx_buffer[1:1 + PACKET_SIZE + DIAG_PACKET_SIZE]
                rx_buffer = rx_buffer[1 + PACKET_SIZE + DIAG_PACKET_SIZE:]
                
                cir_complex = unpack_24bit_signed_vectorized(payload)
                # cir_complex = np.empty(NUM_SAMPLES, dtype=np.complex64)
                # for i in range(NUM_SAMPLES):
                #     offset = i * BYTES_PER_SAMPLE
                #     real = unpack_24bit_signed(payload[offset:offset+3])
                #     imag = unpack_24bit_signed(payload[offset+3:offset+6])
                #     cir_complex[i] = real + 1j * imag
                if points_rx < stopping_points:
                    cir_storage_rx[points_rx] = cir_complex

                diag_payload = payload[PACKET_SIZE:]
                if len(diag_payload) == DIAG_PACKET_SIZE:
                    diag = parse_diag_packet(diag_payload)
                    if diag and points_rx < stopping_points:
                        index_fp_u32_storage_rx[points_rx] = diag['index_fp_u32']
                        accumCount_storage_rx[points_rx] = diag['accumCount']
                        D_storage_rx[points_rx] = diag['D']
                        timestamp_storage_rx[points_rx] = diag['timestamp']
                        p_los_storage_rx[points_rx] = diag['plos']
                        local_host_timestamp_storage_rx[points_rx] = diag['localhost_timestamp']
                        points_rx += 1
                        progress_bar.update(1)
        progress_bar.close()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(f"Total points collected: {points_rx}")
    graceful_exit()

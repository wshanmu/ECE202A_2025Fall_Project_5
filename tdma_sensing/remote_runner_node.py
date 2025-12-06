import serial
import time
import argparse
import numpy as np
import struct
from tqdm import tqdm
import os

# for raspberry pi to send command to UWB as a remote listener
# send command to inialize the UWB
# parser the data from UWB
# send back the data to the localhost

def send_command(channel, cir_sample=28, cir_start_index=0):
    if MODE not in ['TDMA']:
        raise ValueError("Invalid mode. Use 'TDMA'.")
    elif MODE == 'TDMA':
        command = f"TDMA\n"
    CFG_command = f"TCFG -CHAN={channel} -PCODE=10 -CIRSAMPLE={cir_sample} -CIRSI={cir_start_index}\n"
    ser.write(CFG_command.encode())
    time.sleep(0.1)
    ser.write(command.encode())
    print(f"Sent '{MODE}' command to the device.")
    time.sleep(0.5)

def unpack_24bit_signed(data_bytes):
    value = int.from_bytes(data_bytes, byteorder='little', signed=False)
    if value & 0x800000:
        value -= 0x1000000
    return value

def parse_diag_packet(payload):
    if len(payload) != DIAG_PACKET_SIZE:
        print("Invalid packet length")
        ser.write(b"STOP\n")
        return None
    try:
        unpacked = struct.unpack('<IHh7B', payload) # I: uint32_t, H: uint16_t, B: uint8_t
        diag = {
            'index_fp_u32': unpacked[0],
            'accumCount': unpacked[1],
            'plos': unpacked[2],
            'D': unpacked[3],  # frame id
            'sender_ID': unpacked[4],
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

parser = argparse.ArgumentParser(description='CIR Extraction Script')
parser.add_argument('--port', type=str, default='/dev/ttyACM0', help='Serial port to use')
parser.add_argument('--baud', type=int, default=1000000, help='Baud rate for serial communication')
parser.add_argument('--mode', type=str, default='LISN', help='Mode to set for the device')
parser.add_argument('--save', action='store_true', help='Save the data to file')
parser.add_argument('--en', action='store_true', help='whether the tag is enable or not')
parser.add_argument('--suffix', type=str, default='', help='Suffix for filename')
parser.add_argument('--time', type=int, default=10, help='Duration time in seconds')
parser.add_argument('--cir_sample', type=int, default=28, help='CIR sample size')
parser.add_argument('--cir_start_index', type=int, default=0, help='CIR start index')
parser.add_argument('--channel', type=int, default=9, help='Channel number')
parser.add_argument('--fps', type=int, default=1000, help='Frames per second for remote speaker')


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
DIAG_PACKET_SIZE = 10 + 5
SAVING_PATH = './cir_files/'
os.makedirs(SAVING_PATH + 'rx', exist_ok=True)

ser = serial.Serial(PORT, 20, timeout=1)
# ser.set_buffer_size(rx_size=16384, tx_size=4096)
# 192.168.0.131: listener remote IP
duration_time = args.time
frame_rate = args.fps
stopping_points = int(duration_time * frame_rate * 4) # 4 nodes take turns, effectively 4x fps

# RX Storage
cir_storage_rx = np.zeros((stopping_points, NUM_SAMPLES), dtype=np.complex64)
accumCount_storage_rx = np.zeros((stopping_points,), dtype=np.uint32)
index_fp_u32_storage_rx = np.zeros((stopping_points,), dtype=np.uint32)
F1_storage_rx = np.zeros((stopping_points,), dtype=np.uint32)
F2_storage_rx = np.zeros((stopping_points,), dtype=np.uint32)
F3_storage_rx = np.zeros((stopping_points,), dtype=np.uint32)
D_storage_rx = np.zeros((stopping_points,), dtype=np.uint32)
sender_ID_storage_rx = np.zeros((stopping_points,), dtype=np.uint32)
CI_storage_rx = np.zeros((stopping_points,), dtype=np.int32)
p_los_storage_rx = np.zeros((stopping_points,), dtype=np.int16)
CO_storage_rx = np.zeros((stopping_points,), dtype=np.int16)
timestamp_storage_rx = np.zeros((stopping_points,), dtype=np.float64)
local_host_timestamp_storage_rx = np.zeros((stopping_points,), dtype=np.float64)

points_rx = 0

def update():
    global points_rx, start_time

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
            # time_stamp = time.strftime("%Y%m%d_%H%M")
            suffix = '_EN' + SUFFIX if EN else SUFFIX

            # Save RX data
            np.save(f"{SAVING_PATH}/rx/cir_data_rx_{suffix}.npy", cir_storage_rx[:points_rx])
            diag_storage_rx = [{
                'index_fp_u32': index_fp_u32_storage_rx[i],
                'accumCount': accumCount_storage_rx[i],
                'D': D_storage_rx[i],
                'timestamp': timestamp_storage_rx[i],
                'p_los': p_los_storage_rx[i],
                'sender_ID': sender_ID_storage_rx[i],
            } for i in range(points_rx)]
            np.save(f"{SAVING_PATH}/diag/diag_data_rx_{suffix}.npy", diag_storage_rx)
            print("Saved RX data.")
        
    except Exception as e:
        print(f"Error during exit: {e}")


if __name__ == "__main__":
    send_command(CHANNEL, NUM_SAMPLES, CIR_START_INDEX)
    
    rx_buffer = bytearray()
    start_time = time.time()
    end_time = start_time + args.time

    try:

        # # Initialize progress bar
        # progress_bar = tqdm(total=stopping_points, desc="Processing", unit="frames")

        while time.time() < end_time:
            # Read available data from RX and TX
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
                
                # cir_complex = []
                # for i in range(NUM_SAMPLES):
                #     offset = i * BYTES_PER_SAMPLE
                #     real = unpack_24bit_signed(payload[offset:offset+3])
                #     imag = unpack_24bit_signed(payload[offset+3:offset+6])
                #     cir_complex.append(complex(real, imag))
                # cir_array = np.array(cir_complex, dtype=np.complex64)
                cir_complex = unpack_24bit_signed_vectorized(payload)
                if points_rx < stopping_points:
                    cir_storage_rx[points_rx] = cir_complex

                diag_payload = payload[PACKET_SIZE:]
                if len(diag_payload) == DIAG_PACKET_SIZE:
                    diag = parse_diag_packet(diag_payload)
                    if diag and points_rx < stopping_points:
                        index_fp_u32_storage_rx[points_rx] = diag['index_fp_u32']
                        accumCount_storage_rx[points_rx] = diag['accumCount']
                        D_storage_rx[points_rx] = diag['D']
                        sender_ID_storage_rx[points_rx] = diag['sender_ID']
                        timestamp_storage_rx[points_rx] = diag['timestamp']
                        p_los_storage_rx[points_rx] = diag['plos']
                        points_rx += 1
                        # progress_bar.update(1)
        # progress_bar.close()

    except KeyboardInterrupt:
        pass
    print("In total points_rx:", points_rx)
    graceful_exit()
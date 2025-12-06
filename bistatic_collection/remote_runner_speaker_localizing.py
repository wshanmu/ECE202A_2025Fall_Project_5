import serial
import time
import argparse

# Remote runner for speaker node in bistatic localization setup
# Sends SCFG and SPEAKER commands to UWB device

def send_command(mode='SPEAKER', fps=1024, duration=60, channel=9, delayTX=6, xtal_trim=None):
    if mode not in ['SPEAKER']:
        raise ValueError("Invalid mode. Use 'SPEAKER'.")
    if xtal_trim is not None:
        CFG_command = f"SCFG -CHAN={channel} -PCODE=10 -XTALTRIM={xtal_trim} -FPS={fps} -DURATION={duration} -DELAYTX={delayTX}\n"
    else:
        CFG_command = f"SCFG -CHAN={channel} -PCODE=10 -FPS={fps} -DURATION={duration} -DELAYTX={delayTX}\n"
    ser.write(CFG_command.encode())
    time.sleep(0.1)
    
    command = f"{mode}\n"
    ser.write(command.encode())
    print(f"Sent 'SPEAKER' command to the device with channel={channel}, fps={fps}, duration={duration}s, delayTX={delayTX}, xtal_trim={xtal_trim}.")
    time.sleep(0.5)

parser = argparse.ArgumentParser(description='CIR Speaker Script for Localizing')
parser.add_argument('--port', type=str, default='/dev/ttyACM0', help='Serial port to use')
parser.add_argument('--baud', type=int, default=1000000, help='Baud rate for serial communication')
parser.add_argument('--mode', type=str, default='SPEAKER', help='Mode to set for the device')
parser.add_argument('--fps', type=int, default=1024, help='Frames per second')
parser.add_argument('--duration', type=int, default=10, help='Duration in seconds')
parser.add_argument('--channel', type=int, default=9, help='Channel number')
parser.add_argument('--delayTX', type=int, default=6, help='Offset to delay TX transmission')
parser.add_argument('--xtal_trim', type=int, default=None, help='Crystal trim value')

args = parser.parse_args()

# === Configuration ===
PORT = args.port
BAUD = args.baud
MODE = args.mode
FPS = args.fps
DURATION = args.duration
CHANNEL = args.channel
DELAYTX = args.delayTX
XTAL_TRIM = args.xtal_trim

# === Open Serial Port ===
ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"Listening on {PORT}...")
send_command(mode=MODE, fps=FPS, duration=DURATION, channel=CHANNEL, delayTX=DELAYTX, xtal_trim=XTAL_TRIM)

# Wait for the duration to complete
print(f"Speaker will run for {DURATION} seconds...")
time.sleep(DURATION)
print("Speaker duration completed.")

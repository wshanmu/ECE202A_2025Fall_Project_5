from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
import csv
import posixpath
import shlex
import shutil
import time
import paramiko
import yaml
from tqdm import tqdm


@dataclass
class RemoteBoardSession:
    board_id: int
    role: str  # 'LISTENER' or 'SPEAKER'
    config: Dict[str, Optional[str]]
    ssh: paramiko.SSHClient
    stdin: Any
    stdout: Any
    stderr: Any
    channel: paramiko.Channel
    remote_suffix: str


def get_latest_remote_file(sftp: paramiko.SFTPClient, remote_dir: str) -> Optional[str]:
    """Get the most recently modified .npy file in a remote directory."""
    files = [f for f in sftp.listdir(remote_dir) if f.endswith('.npy')]
    if not files:
        return None
    latest_file = max(
        files,
        key=lambda name: sftp.stat(posixpath.join(remote_dir, name)).st_mtime,
    )
    return latest_file


# Board configuration for bistatic localization
# Node 0: SPEAKER (transmitter)
# Nodes 1, 2, 3: LISTENER (receivers)
BOARD_CONFIG: Dict[int, Dict[str, Optional[str]]] = {
    0: {
        "ip": "192.168.0.115",
        "username": "icon",
        "password": "icon",
        "serial_port": "/dev/ttyACM0",
        "remote_base_dir": "/home/icon/UWB_code",
        "python_bin": "python3",
        "script_path": "remote_runner_speaker_localizing.py",
        "usb_hub": "1-1",
        "role": "SPEAKER",
        # "role": "LISTENER",
    },
    1: {
        "ip": "192.168.0.120",
        "username": "icon",
        "password": "icon",
        "serial_port": "/dev/ttyACM0",
        "remote_base_dir": "/home/icon/uwb",
        "python_bin": "./venv/bin/python",
        # "script_path": "remote_runner_listener_localizing.py",
        "script_path": "remote_runner_speaker_localizing.py",
        "usb_hub": "3",
        "role": "SPEAKER",
        # "role": "LISTENER",
    },
    2: {
        "ip": "192.168.0.46",
        "username": "icon",
        "password": "icon",
        "serial_port": "/dev/ttyACM0",
        "remote_base_dir": "/home/icon/uwb",
        "python_bin": "./venv/bin/python",
        "script_path": "remote_runner_listener_localizing.py",
        "usb_hub": "3",
        "role": "LISTENER",
    },
    3: {
        "ip": "192.168.0.97",
        "username": "icon",
        "password": "icon",
        "serial_port": "/dev/ttyACM0",
        "remote_base_dir": "/home/icon/uwb",
        "python_bin": "./venv/bin/python",
        "script_path": "remote_runner_listener_localizing.py",
        "usb_hub": "3",
        "role": "LISTENER",
    },
}


LOCAL_CIR_BASE = Path(__file__).resolve().parent / "cir_files"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config_yaml(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse configuration values from the nested YAML structure."""
    radios = config.get('radios', [])
    rooms = config.get('rooms', [])
    
    # Extract values from lists of single-key dictionaries
    radio_params = {}
    for item in radios:
        radio_params.update(item)
    
    room_params = {}
    for item in rooms:
        room_params.update(item)
    
    return {
        'channel': radio_params.get('channel', 5),
        'cir_sample': radio_params.get('cir_sample', 28),
        'cir_starting_idx': radio_params.get('cir_starting_idx', 0),
        'fps': radio_params.get('fps', 1000),
        'duration': radio_params.get('duration', 10),
        'delayTX': radio_params.get('delayTX', 0),
        'xtal_trim': radio_params.get('xtal_trim', 25),
        'room': room_params.get('room', ''),
        'desk_num': room_params.get('desk_num', 0),
        'desk_layout': room_params.get('desk_layout', ''),
        'humans': room_params.get('humans', 0),
        'activity': room_params.get('activity', ''),
        'tag_position': room_params.get('tag_position', ''),
    }


def compose_remote_suffix(user_suffix: str, board_id: int) -> str:
    """Create suffix for remote files with node ID."""
    node_suffix = f"_node{board_id}"
    return f"{user_suffix}{node_suffix}" if user_suffix else node_suffix


def build_listener_command(
    cfg: Dict[str, Optional[str]],
    duration: int,
    channel: int,
    fps: int,
    cir_sample: int,
    cir_start_index: int,
    save: bool,
    en: bool,
    suffix: str,
) -> str:
    """Build command for listener nodes."""
    if not cfg.get("remote_base_dir"):
        raise ValueError("Remote base directory is not configured.")
    if not cfg.get("python_bin"):
        raise ValueError("Python interpreter path is not configured.")
    if not cfg.get("script_path"):
        raise ValueError("Remote script path is not configured.")
    if not cfg.get("serial_port"):
        raise ValueError("Serial port is not configured.")

    parts: List[str] = [
        "cd",
        shlex.quote(cfg["remote_base_dir"]),
        "&&",
        shlex.quote(cfg["python_bin"]),
        shlex.quote(cfg["script_path"]),
        "--mode",
        "LISTENER",
        "--port",
        shlex.quote(cfg["serial_port"]),
        "--time",
        str(duration),
        "--channel",
        str(channel),
        "--fps",
        str(fps),
        "--cir_sample",
        str(cir_sample),
        "--cir_start_index",
        str(cir_start_index),
    ]

    if save:
        parts.append("--save")
    if en:
        parts.append("--en")
    if suffix:
        parts.extend(["--suffix", shlex.quote(suffix)])

    return " ".join(parts)


def build_speaker_command(
    cfg: Dict[str, Optional[str]],
    duration: int,
    channel: int,
    fps: int,
    delayTX: int,
    xtal_trim: int,
) -> str:
    """Build command for speaker node."""
    if not cfg.get("remote_base_dir"):
        raise ValueError("Remote base directory is not configured.")
    if not cfg.get("python_bin"):
        raise ValueError("Python interpreter path is not configured.")
    if not cfg.get("script_path"):
        raise ValueError("Remote script path is not configured.")
    if not cfg.get("serial_port"):
        raise ValueError("Serial port is not configured.")

    parts: List[str] = [
        "cd",
        shlex.quote(cfg["remote_base_dir"]),
        "&&",
        shlex.quote(cfg["python_bin"]),
        shlex.quote(cfg["script_path"]),
        "--mode",
        "SPEAKER",
        "--port",
        shlex.quote(cfg["serial_port"]),
        "--duration",
        str(duration),
        "--channel",
        str(channel),
        "--fps",
        str(fps),
        "--delayTX",
        str(delayTX),
        "--xtal_trim",
        str(xtal_trim),
    ]

    return " ".join(parts)


def start_remote_board(
    board_id: int,
    duration: int,
    channel: int,
    fps: int,
    cir_sample: int,
    cir_start_index: int,
    delayTX: int,
    xtal_trim: int,
    save: bool,
    en: bool,
    suffix: str,
) -> RemoteBoardSession:
    """Start a remote board session (listener or speaker)."""
    cfg = BOARD_CONFIG.get(board_id)
    if cfg is None:
        raise ValueError(f"Board {board_id} is not defined.")
    ip = cfg.get("ip")
    if not ip:
        raise ValueError(f"Board {board_id} does not have an IP configured.")
    
    role = cfg.get("role", "LISTENER")

    print(f"[Board {board_id} - {role}] Connecting to {ip} as {cfg['username']}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username=cfg.get("username"), password=cfg.get("password"))

    if role == "SPEAKER":
        command = build_speaker_command(
            cfg=cfg,
            duration=duration,
            channel=channel,
            fps=fps,
            delayTX=delayTX,
            xtal_trim=xtal_trim,
        )
    else:  # LISTENER
        command = build_listener_command(
            cfg=cfg,
            duration=duration,
            channel=channel,
            fps=fps,
            cir_sample=cir_sample,
            cir_start_index=cir_start_index,
            save=save,
            en=en,
            suffix=suffix,
        )

    print(f"[Board {board_id} - {role}] Starting remote runner: {command}")
    stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
    channel_obj = stdout.channel
    return RemoteBoardSession(
        board_id=board_id,
        role=role,
        config=cfg,
        ssh=ssh,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        channel=channel_obj,
        remote_suffix=suffix,
    )


def wait_for_completion(sessions: List[RemoteBoardSession], duration: int) -> None:
    """Wait for all sessions to complete within the specified duration."""
    start_time = time.time()
    target_end = start_time + duration
    bar = tqdm(total=duration, desc="Bistatic Capture", unit="s")
    last_reported = 0

    try:
        while True:
            if all(session.channel.exit_status_ready() for session in sessions):
                break
            now = time.time()
            elapsed = now - start_time
            reported = min(int(elapsed), duration)
            if reported > last_reported:
                bar.update(reported - last_reported)
                last_reported = reported
            if now >= target_end:
                break
            time.sleep(0.3)
    finally:
        if last_reported < duration:
            bar.update(duration - last_reported)
        bar.close()

    time.sleep(1.5)  # Wait a bit before forcing shutdown
    for session in sessions:
        if not session.channel.exit_status_ready():
            print(f"[Board {session.board_id}] Forcing shutdown after duration exceeded...")
            stop_remote_session(session)


def stop_remote_session(session: RemoteBoardSession) -> None:
    """Stop a remote session by sending Ctrl+C."""
    if session.channel.exit_status_ready():
        return
    try:
        time.sleep(2.0)
        session.channel.send("\x03")
    except Exception as exc:
        print(f"[Board {session.board_id}] Failed to send Ctrl+C: {exc}")
    while not session.channel.exit_status_ready():
        time.sleep(0.2)


def collect_remote_output(session: RemoteBoardSession) -> None:
    """Collect and print output from remote session."""
    try:
        stdout_data = session.stdout.read()
        stderr_data = session.stderr.read()
    except Exception as exc:
        print(f"[Board {session.board_id}] Unable to read remote output: {exc}")
        return

    if stdout_data:
        print(f"[Board {session.board_id}] Output:\n{stdout_data.strip()}")
    if stderr_data:
        print(f"[Board {session.board_id}] Error:\n{stderr_data.strip()}")


def retrieve_latest_files(session: RemoteBoardSession, local_output_dir: Path) -> None:
    """Retrieve latest CIR files from remote board and save to local directory."""
    # Only retrieve files from listener nodes
    if session.role != "LISTENER":
        print(f"[Board {session.board_id}] Skipping file retrieval (SPEAKER node)")
        return
    
    base_dir = session.config.get("remote_base_dir")
    if not base_dir:
        print(f"[Board {session.board_id}] Remote base directory not set; skipping download.")
        return

    sftp = session.ssh.open_sftp()
    try:
        for subdir in ("diag", "rx"):
            remote_dir = posixpath.join(base_dir, "cir_files", subdir)
            try:
                latest_file = get_latest_remote_file(sftp, remote_dir)
            except FileNotFoundError:
                print(f"[Board {session.board_id}] Remote directory not found: {remote_dir}")
                continue
            if latest_file is None:
                print(f"[Board {session.board_id}] No .npy files found in {remote_dir}")
                continue

            remote_path = posixpath.join(remote_dir, latest_file)
            local_dir = local_output_dir / subdir
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / latest_file

            print(f"[Board {session.board_id}] Downloading {remote_path} -> {local_path}")
            sftp.get(remote_path, str(local_path))
            try:
                sftp.remove(remote_path)
                print(f"[Board {session.board_id}] Removed remote file: {remote_path}")
            except IOError:
                print(f"[Board {session.board_id}] Could not remove remote file: {remote_path}")
    finally:
        sftp.close()


def power_cycle_device(session: RemoteBoardSession) -> None:
    """Power cycle USB hub to reset device."""
    hub = session.config.get("usb_hub")
    if not hub:
        return
    try:
        print(f"[Board {session.board_id}] Power cycling USB hub {hub}...")
        if hub == "3":
            # Raspberry Pi 5
            session.ssh.exec_command(f"sudo uhubctl -l 3 -a 0")
            session.ssh.exec_command(f"sudo uhubctl -l 1 -a 0")
            time.sleep(1)
            session.ssh.exec_command(f"sudo uhubctl -l 3 -a 1")
            session.ssh.exec_command(f"sudo uhubctl -l 1 -a 1")
            time.sleep(2)
        else:
            # Other Raspberry Pi models
            session.ssh.exec_command(f"sudo uhubctl -l {hub} -a 0")
            time.sleep(1)
            session.ssh.exec_command(f"sudo uhubctl -l {hub} -a 1")
            time.sleep(2)
        print(f"[Board {session.board_id}] Power cycle completed.")
    except Exception as exc:
        print(f"[Board {session.board_id}] USB power cycle failed: {exc}")


def create_output_directory(base_dir: Path, suffix: str) -> Path:
    """Create a timestamped output directory for this experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{suffix}" if suffix else timestamp
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def copy_config_to_output(config_path: Path, output_dir: Path) -> None:
    """Copy the configuration YAML file to the output directory."""
    dest_path = output_dir / config_path.name
    shutil.copy2(config_path, dest_path)
    print(f"Copied config file to {dest_path}")


def update_csv_summary(csv_path: Path, config_params: Dict[str, Any], output_dir: Path) -> None:
    """Update or create CSV summary file with experiment information."""
    csv_exists = csv_path.exists()
    
    # Define CSV columns
    fieldnames = [
        'timestamp',
        'output_dir',
        'channel',
        'cir_sample',
        'cir_starting_idx',
        'fps',
        'duration',
        'delayTX',
        'xtal_trim',
        'room',
        'desk_num',
        'desk_layout',
        'humans',
        'activity',
        'tag_position',
    ]
    
    # Prepare row data
    row_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'output_dir': output_dir.name,
        'channel': config_params.get('channel', ''),
        'cir_sample': config_params.get('cir_sample', ''),
        'cir_starting_idx': config_params.get('cir_starting_idx', ''),
        'fps': config_params.get('fps', ''),
        'duration': config_params.get('duration', ''),
        'delayTX': config_params.get('delayTX', ''),
        'xtal_trim': config_params.get('xtal_trim', ''),
        'room': config_params.get('room', ''),
        'desk_num': config_params.get('desk_num', ''),
        'desk_layout': config_params.get('desk_layout', ''),
        'humans': config_params.get('humans', ''),
        'activity': config_params.get('activity', ''),
        'tag_position': config_params.get('tag_position', ''),
    }
    
    # Write to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new or empty
        if not csv_exists or csvfile.tell() == 0:
            writer.writeheader()
        
        writer.writerow(row_data)
    
    print(f"Updated CSV summary at {csv_path}")


parser = argparse.ArgumentParser(description="Bistatic CIR localization controller")
parser.add_argument(
    "--config",
    type=str,
    default="cir_localizing.yaml",
    help="Path to the configuration YAML file (default: cir_localizing.yaml)",
)
parser.add_argument(
    "--save",
    action="store_true",
    help="Save remote captures (always enabled for listeners)",
)
parser.add_argument("--en", action="store_true", help="Enable the tag on the remote devices")
parser.add_argument("--suffix", type=str, default="", help="Suffix to append before the node identifier")
parser.add_argument("--time", type=int, default=None, help="Duration time in seconds (overrides config)")
parser.add_argument("--fps", type=int, default=None, help="Frames per second (overrides config)")
parser.add_argument("--channel", type=int, default=None, help="Channel number (overrides config)")
parser.add_argument("--cir_sample", type=int, default=None, help="CIR sample size (overrides config)")
parser.add_argument(
    "--cir_start_index",
    type=int,
    default=None,
    help="CIR start index (overrides config)",
)
parser.add_argument("--delayTX", type=int, default=None, help="TX delay offset (overrides config)")
parser.add_argument("--xtal_trim", type=int, default=None, help="Crystal trim value (overrides config)")
parser.add_argument(
    "--boards",
    type=int,
    nargs="*",
    help="Board IDs to trigger. Defaults to all configured boards (0, 1, 2, 3)",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load configuration from YAML file
    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = CONFIG_DIR / config_file
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    print(f"Loading configuration from {config_file}")
    yaml_config = load_config_yaml(config_file)
    config_params = parse_config_values(yaml_config)
    
    # Command-line arguments override config file values
    duration = args.time if args.time is not None else config_params['duration']
    fps = args.fps if args.fps is not None else config_params['fps']
    channel = args.channel if args.channel is not None else config_params['channel']
    cir_sample = args.cir_sample if args.cir_sample is not None else config_params['cir_sample']
    cir_start_index = args.cir_start_index if args.cir_start_index is not None else config_params['cir_starting_idx']
    delayTX = args.delayTX if args.delayTX is not None else config_params['delayTX']
    xtal_trim = args.xtal_trim if args.xtal_trim is not None else config_params['xtal_trim']
    
    print(f"Experiment parameters: channel={channel}, fps={fps}, cir_sample={cir_sample}, "
          f"cir_start_index={cir_start_index}, duration={duration}s, delayTX={delayTX}, xtal_trim={xtal_trim}")

    # Create output directory with timestamp and suffix
    output_dir = create_output_directory(LOCAL_CIR_BASE, args.suffix)
    print(f"Output directory: {output_dir}")
    
    # Copy config YAML to output directory
    copy_config_to_output(config_file, output_dir)
    
    # Update CSV summary
    csv_summary_path = LOCAL_CIR_BASE / "experiment_summary_localizing.csv"
    update_csv_summary(csv_summary_path, config_params, output_dir)

    # Select boards to use
    selected_boards = args.boards
    if not selected_boards:
        # Default: use all configured boards
        selected_boards = [0, 1, 2, 3]

    if not selected_boards:
        raise ValueError("No boards selected. Use --boards to specify board IDs.")

    # Separate listeners and speakers
    listener_boards = [bid for bid in selected_boards if BOARD_CONFIG.get(bid, {}).get("role") == "LISTENER"]
    speaker_boards = [bid for bid in selected_boards if BOARD_CONFIG.get(bid, {}).get("role") == "SPEAKER"]

    print(f"Listener boards: {listener_boards}")
    print(f"Speaker boards: {speaker_boards}")

    sessions: List[RemoteBoardSession] = []
    try:
        # Step 1: Start listener nodes (1, 2, 3)
        print("\n=== Starting Listener Nodes ===")
        for board_id in listener_boards:
            suffix = compose_remote_suffix(args.suffix, board_id)
            session = start_remote_board(
                board_id=board_id,
                duration=duration,
                channel=channel,
                fps=fps,
                cir_sample=cir_sample,
                cir_start_index=cir_start_index,
                delayTX=delayTX,
                xtal_trim=xtal_trim,
                save=True,  # Always save for listeners
                en=args.en,
                suffix=suffix,
            )
            sessions.append(session)
            time.sleep(0.5)  # Small delay between starting nodes

        # Wait a bit for listeners to initialize
        print("Waiting for listeners to initialize...")
        time.sleep(2)

        # Step 2: Start speaker node (0)
        print("\n=== Starting Speaker Node ===")
        for board_id in speaker_boards:
            suffix = compose_remote_suffix(args.suffix, board_id)
            session = start_remote_board(
                board_id=board_id,
                duration=duration,
                channel=channel,
                fps=fps,
                cir_sample=cir_sample,
                cir_start_index=cir_start_index,
                delayTX=delayTX,
                xtal_trim=xtal_trim,
                save=False,  # Speaker doesn't need to save
                en=args.en,
                suffix=suffix,
            )
            sessions.append(session)

        # Step 3: Wait for completion
        print("\n=== Waiting for Completion ===")
        wait_for_completion(sessions, duration)

    except KeyboardInterrupt:
        print("\nInterrupted locally. Stopping remote sessions...")
        time.sleep(1.0)
        for session in sessions:
            stop_remote_session(session)
    finally:
        # Collect outputs
        print("\n=== Collecting Remote Outputs ===")
        for session in sessions:
            collect_remote_output(session)

        # Retrieve files from listener nodes only
        print("\n=== Retrieving Files from Listeners ===")
        for session in sessions:
            if session.role == "LISTENER":
                retrieve_latest_files(session, output_dir)

        # Power cycle all devices
        print("\n=== Power Cycling Devices ===")
        for session in sessions:
            power_cycle_device(session)

        # Close all SSH connections
        print("\n=== Closing SSH Connections ===")
        for session in sessions:
            try:
                session.ssh.close()
                print(f"[Board {session.board_id}] SSH connection closed.")
            except Exception as exc:
                print(f"[Board {session.board_id}] Error closing SSH: {exc}")

        print(f"\nExperiment completed. Data saved to: {output_dir}")

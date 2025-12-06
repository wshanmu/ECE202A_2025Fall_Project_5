import subprocess


tx_trim = {
    'tag 0': 31,
    'tag 1': 22,
    'tag 2': 25,
    'tag 3': 25,
}

# Define common command components
python_exe = "/home/shanmu/miniconda3/envs/py10/bin/python"
script_path = "./bistatic_collection/cir_control_remote_nodes_localizing.py"
fps = 1024
duration = 50
for delay_tx in [6]:
    # Loop through exp1 to exp10
    for i in range(1, 3):
        for tx_xtal_trim in [22]:
            suffix_base = f"boelterOneLongVer_tag1234_TX1_RX3_14bit_24cir_ch9_xtalTX{tx_xtal_trim}_delay{delay_tx}_exp"
            suffix = f"{suffix_base}{i}"
            cmd = [
                python_exe,
                script_path,
                "--fps", str(fps),
                "--time", str(duration),
                "--delayTX", str(delay_tx),
                "--suffix", suffix,
                '--channel', '9',
                '--xtal_trim', str(tx_xtal_trim),
                "--boards", "3", "1",
                "--cir_sample", "24",
            ]
            print(f"Running experiment {i}...")
            subprocess.run(cmd)

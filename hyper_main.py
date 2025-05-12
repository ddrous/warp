## Use this script to run several experiments in a row.

import os

configs = ["msd_2", "msd", "sine_huge", "sine_large", "sine_small", "sine_medium", "sine_tiny"]

for config in configs:
    print(f"\n\n=========== Running {config}... ===========", flush=True)

    os.system(f"python main.py cfgs/wsm/{config}.yaml")

    print(f"Finished running {config}.", flush=True)

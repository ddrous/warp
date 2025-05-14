## Use this script to run several experiments in a row.

import os

configs = ["electricity_h1", "electricity_h2"]

for config in configs:
    print(f"\n\n=========== Running {config}... ===========", flush=True)

    os.system(f"python main.py cfgs/wsm/{config}.yaml")

    print(f"Finished running {config}.", flush=True)

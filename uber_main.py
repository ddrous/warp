## Use this script to run several experiments in a row.

import os

configs = ["small"]

for config in configs:
    print(f"\n\n=========== Running {config}... ===========", flush=True)

    os.system(f"python main.py cfgs/wsm/sine/sine_{config}.yaml")

    print(f"Finished running {config}.", flush=True)

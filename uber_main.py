## Use this script to run several experiments in a row.

import os

# configs = ["small"]
configs = ["cfgs/wsm/celeba/celeba.yaml"]

for config in configs:
    print(f"\n\n=========== Running {config}... ===========", flush=True)

    # os.system(f"python main.py cfgs/wsm/sine/sine_{config}.yaml")
    os.system(f"python main.py {config}")

    print(f"Finished running {config}.", flush=True)

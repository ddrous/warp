## Use this script to run several experiments in a row.

import os

# suffixes = ["pe", "seqlen64", "baseline1", "baseline2", "baseline3", "baseline4", "deeproot", "smallbatch", "largebatch", "reluactivation", "shallowinitnet"]
suffixes = ["seqlen512"]
configs = [f"cfgs/wsm/icl/experiments/{suffix}.yaml" for suffix in suffixes]

for config in configs:
    print(f"\n\n=========== Running {config}... ===========", flush=True)

    os.system(f"python main.py {config}")

    print(f"Finished running {config}.", flush=True)

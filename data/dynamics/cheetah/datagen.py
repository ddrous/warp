#%%
## Code to visualize the Cheetah dataset

import pandas as pd
import os
import numpy as np


## Plots all 17 dimensions for all 1000 time steps, all 31 traces
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


os.system("ls -a")

traces = []

for i in range(31):
    traces.append(np.load(f"raw/trace_00{i:02d}.npy"))

traces = np.stack(traces)
# print(trace0.shape)

# ### Cropp everythin between -1 and 1
# traces = np.clip(traces, -1, 1)

## Split the data into overlapping chunks of 32
chunk_size = 32
overlap = 2
new_traces = []
for trace in traces:
    for i in range(0, len(trace) - chunk_size + 1, chunk_size - overlap):
        new_traces.append(trace[i:i + chunk_size])
new_traces = np.stack(new_traces)
print(new_traces.shape)


# # Use the funciton below to split the data into overlapping chunks of 32
# def cut_in_sequences(x, seq_len, inc=1):
#     """
#     Cut the input array into overlapping sequences.
#     """
#     sequences_x = []
#     print("x.shape", x.shape)
#     for s in range(0, x.shape[0]-seq_len-1, inc):
#         start = s
#         end = start + seq_len
#         sequences_x.append(x[:, start:end])

#     print(f"Cut in sequences: {len(sequences_x)}")
#     return np.stack(sequences_x, axis=1)

# new_traces = cut_in_sequences(traces, 32, 10)

traces = new_traces

np.save("test.npy", traces[5:15])
np.save("train.npy", traces[15:25])

plt.figure(figsize=(20, 10))
plt.title("Cheetah traces")
plt.xlabel("Time step")
plt.ylabel("Value")

# for i in range(1):
#     plt.subplot(5, 4, i + 1)
#     plt.title("Dimension " + str(i))
#     plt.xlabel("Time step")
#     plt.ylabel("Value")
#     for j in range(31):
#         # plt.plot(traces[j, :, i], alpha=0.5)
#         plt.plot(traces[j,:,i], alpha=0.5, label="Trace " + str(j))
#         plt.legend()

# start = np.random.randint(0, 31, size=1)[0]
# stop = np.random.randint(start+1, 31, size=1)[0]
start, stop = 0, 2

for trace in range(31):
    plt.subplot(6, 6, trace + 1)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.plot(traces[trace, :, start:stop], alpha=0.5)
    # plt.legend()
    plt.title("Trace " + str(trace))
    plt.xlabel("Time step")

plt.tight_layout()
plt.show()
# plt.savefig("cheetah_traces.png")


#%%

## print the mean and std along each dimension
means = np.mean(traces, axis=(0,1))
stds = np.std(traces, axis=(0,1))
print(f"Means:", means)
print(f"Stds:", stds)

### Plot the means and stds
plt.figure(figsize=(20, 10))
plt.title("Cheetah traces means and stds")
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.subplot(1, 2, 1)

plt.plot(means, alpha=0.5)
plt.title("Means")
plt.xlabel("Dimension")
plt.subplot(1, 2, 2)
plt.plot(stds, alpha=0.5)
plt.title("Stds")
plt.xlabel("Dimension")

#%%
## 
## Print Min and Max
print("Min:", np.min(traces))
print("Max:", np.max(traces))

## Plot this against a normal tanh
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
x = np.linspace(-30.5, 30.5, 500)
y = np.tanh(x)
a, b, alpha, beta = 10, 0, 20, 0
y2 = alpha * np.tanh((x-b)/a) + beta
ax.plot(x, y, label="tanh")
ax.plot(x, y2, label="Dynamic tanh")
ax.set_title("Dynamic tanh vs tanh after training")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.draw();
# plt.savefig("dynamic_tanh.png", dpi=100, bbox_inches='tight')






#%%



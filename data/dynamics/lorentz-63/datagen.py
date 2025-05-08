#%%

## Load the noisy.pt
import torch

data = torch.load("noisy.pt")
print(data.shape)

## This is a Lorentz-63 system. Plot thei first 100 steps of one variable against timeanother
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

dim0, dim1 = 0, 1
fig, ax = plt.subplots(8, 8, figsize=(4*8, 4*8))
ax = ax.flatten()

for i in range(64):
    ax[i].plot(data[i, :1000, dim0], data[i, :1000, dim1], markersize=1)
    # ax[i].set_xlabel("x")
    # ax[i].set_ylabel("y")

plt.tight_layout()
plt.show()


## Save as train,npy
train = data[:48, :1000, :]
np.save("train.npy", train)
## Save as test.npy
test = data[48:, :1000, :]
np.save("test.npy", test)

print("Min and max of train: ", train.min(), train.max())

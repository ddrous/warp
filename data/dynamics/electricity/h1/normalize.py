#%%

# Create the folder called "old" and copy the train test in there
import os

if not os.path.exists("old"):
    os.makedirs("old")
    os.system("cp train.npy old/train.npy")
    os.system("cp test.npy old/test.npy")

# load the data
import numpy as np
train = np.load("old/train.npy")
test = np.load("old/test.npy")

## Plot all features of the first sample in the train set (on the same plot)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
for i in range(train.shape[2]):
    plt.plot(train[70, :, i], label=f"Feature {i}")
plt.title("All features of the first sample in the train set")
plt.xlabel("Time step")



## each file in the format (n_samples, seq_len, n_features). I want to normalize the features independently so that they end up in the range (-1, 1). The test uses the same stats as the train.

min_train = np.min(train, axis=(0, 1))
max_train = np.max(train, axis=(0, 1))

# Normalize the train data
train = (train - min_train) / (max_train - min_train)
train = train * 2 - 1
# Normalize the test data
test = (test - min_train) / (max_train - min_train)
test = test * 2 - 1

# Save the data
np.save("train.npy", train)
np.save("test.npy", test)



plt.figure(figsize=(10, 5))
for i in range(train.shape[2]):
    plt.plot(train[70, :, i], label=f"Feature {i}")
plt.title("All features of the first sample in the train set")
plt.xlabel("Time step")
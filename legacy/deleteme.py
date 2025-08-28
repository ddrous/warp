#%%
## Test the plot positional encoding with:
# sequence length (L) = 256
# batch size (B) = 1
# embedding dimension (D) = 2
Cte = 5**1

import matplotlib.pyplot as plt
import numpy as np
def plot_positional_encoding(L, B, D):
    # Create a positional encoding matrix
    pos_enc = np.zeros((L, D))
    for pos in range(L):
        for i in range(0, D, 2):
            pos_enc[pos, i] = np.sin(pos / (Cte ** (2*i / D)))
            if i + 1 < D:
                pos_enc[pos, i + 1] = np.cos(pos / (Cte ** (2*i / D)))

    # Plot the positional encoding
    plt.figure(figsize=(10, 5))
    plt.imshow(pos_enc.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Positional Encoding')
    plt.xlabel('Position')
    plt.ylabel('Embedding Dimension')
    plt.show()

# Example usage
plot_positional_encoding(L=6, B=1, D=10)
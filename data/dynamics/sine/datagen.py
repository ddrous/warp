#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="notebook")


def generate_sine_waves(nb_samples, seq_len, frequency=5, save_path='sine_waves.npy'):
    """
    Generate sine waves with the same frequency but different phase shifts.
    
    Parameters:
    - nb_samples: Number of different sine waves (with different phase shifts)
    - seq_len: Number of time steps for each sine wave
    - frequency: Frequency of the sine waves (same for all)
    - save_path: Path to save the NumPy array
    
    Returns:
    - trajectories: NumPy array of shape (nb_samples, seq_len, 1)
    """
    # Time vector from 0 to 1
    t = np.linspace(0, 1, seq_len)
    
    # Generate random phase shifts between 0 and 2π
    # phase_shifts = np.random.uniform(0, 2 * np.pi, size=nb_samples)
    phase_shifts = np.random.uniform(-np.pi/6, np.pi/6, size=nb_samples)

    # Initialize trajectories array
    trajectories = np.zeros((nb_samples, seq_len, 1))

    # Generate sine waves with different phase shifts
    for i in range(nb_samples):
        trajectories[i, :, 0] = np.sin(2 * np.pi * frequency * t + phase_shifts[i])

    # Save to NPY file
    np.save(save_path, trajectories)
    
    print(f"Generated {nb_samples} sine waves with shape {trajectories.shape}")
    print(f"Saved to {save_path}")
    
    return trajectories

def visualize_sine_waves(trajectories, max_display=5):
    """
    Visualize a subset of the generated sine waves.
    
    Parameters:
    - trajectories: NumPy array of shape (nb_samples, seq_len, 1)
    - max_display: Maximum number of sine waves to display
    """
    nb_samples = trajectories.shape[0]
    seq_len = trajectories.shape[1]
    
    # Time vector
    t = np.linspace(0, 1, seq_len)
    
    # Display at most max_display sine waves
    n_display = min(nb_samples, max_display)
    
    plt.figure(figsize=(10, 6))
    for i in range(n_display):
        plt.plot(t, trajectories[i, :, 0], label=f"Wave {i+1}")
    
    plt.title('Sine Waves with Different Phase Shifts')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    # plt.savefig('sine_waves.png', dpi=300)
    plt.show()


# Example usage
if __name__ == "__main__":
    # split = "test"

    # seq_len = 32       # Number of time steps
    # frequency = 1       # Frequency of the sine waves

    seq_len = 16       # Number of time steps
    frequency = 1.0       # Frequency of the sine waves

    for split in ["train", "val", "test"]: 
        if split == "train":
            # Parameters
            nb_samples = 10000     # Number of different sine waves
            save_path = 'train.npy'
            np.random.seed(10)

        elif split == "val":
            # Parameters for validation
            nb_samples = 1000
            save_path = 'val.npy'
            np.random.seed(42)
        elif split == "test":
            # Parameters for testing
            nb_samples = 1000
            save_path = 'test.npy'
            np.random.seed(2024)
        else:
            raise ValueError("Invalid split. Choose from 'train', 'val', or 'test'.")

        # Generate sine waves
        trajectories = generate_sine_waves(nb_samples, seq_len, frequency, save_path)

        # Visualize
        visualize_sine_waves(trajectories)
        
        # Example of how to load the saved data
        loaded_trajectories = np.load(save_path)
        print(f"Loaded trajectories shape: {loaded_trajectories.shape}")


#%% Create four folders and put the train data in them
import os
import shutil


if __name__ == "__main__":
    folders = ['tiny', 'small', 'medium', 'large', 'huge']  
    nb_train_samples = [1, 10, 100, 1000, 10000]  # Number of samples for each folder

    for folder, nb_samples in zip(folders, nb_train_samples):
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

        # os.system("ls")
        # Load the train data
        train_data = np.load('train.npy')

        # Select and save the first nb_samples samples
        np.save(os.path.join(folder, 'train.npy'), train_data[:nb_samples])

        # Copy the val and test data
        shutil.copy('val.npy', os.path.join(folder, 'val.npy'))
        shutil.copy('test.npy', os.path.join(folder, 'test.npy'))


#%%
if __name__ == "__main__":
    ## Delete the residual files
    os.remove('train.npy')
    os.remove('val.npy')
    os.remove('test.npy')
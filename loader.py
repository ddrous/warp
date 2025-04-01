#%%
import numpy as np
import torch
import torchvision
from torchvision import transforms
from selfmod import NumpyLoader

class TimeSeriesDataset:
    """
    For any time series dataset, from which the others will inherit
    """
    def __init__(self, dataset, labels, t_eval, traj_prop=1.0):

        self.dataset = dataset
        n_envs, n_timesteps, n_dimensions = dataset.shape
        self.t_eval = t_eval
        self.total_envs = n_envs

        self.labels = labels

        if traj_prop < 0 or traj_prop > 1:
            raise ValueError("The smallest proportion of the trajectory to use must be between 0 and 1")
        self.traj_prop = traj_prop
        self.traj_len = int(n_timesteps * traj_prop)

        self.num_steps = n_timesteps
        self.data_size = n_dimensions

    def __getitem__(self, idx):
        inputs = self.dataset[idx, :, :]
        outputs = self.labels[idx]
        t_eval = self.t_eval
        traj_len = self.traj_len

        if self.traj_prop == 1.0:
            ### STRAIGHFORWARD APPROACH ###
            return (inputs, t_eval), outputs
        else:
            ### SAMPLING APPROACH ###
            ## Select a random trajectory of length t-2
            # start_idx = np.random.randint(0, self.num_steps - traj_len)
            # end_idx = start_idx + traj_len
            # ts = t_eval[start_idx:end_idx]
            # trajs = inputs[start_idx:end_idx, :]
            # return (trajs, ts), outputs

            ## Select a random subset of traj_len-2 indices, then concatenate the start and end points
            indices = np.sort(np.random.choice(np.arange(1,self.num_steps-1), traj_len-2, replace=False))
            indices = np.concatenate(([0], indices, [self.num_steps-1]))
            ts = t_eval[indices]
            trajs = inputs[indices, :]
            return (trajs, ts), outputs

    def __len__(self):
        return self.total_envs


class TrendsDataset(TimeSeriesDataset):
    """
    For the synthetic control dataset from Time Series Classification
    """

    def __init__(self, data_dir, skip_steps=1, traj_prop=1.0):
        try:
            time_series = []
            with open(data_dir+"synthetic_control.data", 'r') as f:
                for line in f:
                    time_series.append(list(map(float, line.split())))
            raw_data = np.array(time_series, dtype=np.float32)
            raw_data = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)        ## normalise the dataset
        except:
            raise ValueError(f"Data not found at {data_dir}")

        dataset = raw_data[:, ::skip_steps, None]

        n_envs, n_timesteps, n_dimensions = dataset.shape

        ## Duplicate t_eval for each environment
        t_eval = np.linspace(0, 1., n_timesteps)

        ## We have 600 samples and 6 classes as above. Create the labels
        labels = np.zeros((600,), dtype=int)
        labels[100:200] = 1 
        labels[200:300] = 2
        labels[300:400] = 3
        labels[400:500] = 4
        labels[500:600] = 5

        self.total_envs = n_envs
        self.nb_classes = 6
        self.num_steps = n_timesteps
        self.data_size = n_dimensions

        super().__init__(dataset, labels, t_eval, traj_prop)



class DynamicsDataset(TimeSeriesDataset):
    """
    For the synthetic control dataset from Time Series Classification
    """

    def __init__(self, data_dir, traj_length=1000):
        try:
            raw_data = torch.load(data_dir)
            # raw_t_eval = np.linspace(0, 1., raw_data.shape[1])
        except:
            raise ValueError(f"Data not loadable at {data_dir}")

        ## Normalise the dataset between 0 and 1
        # raw_data = (raw_data - torch.mean(raw_data)) / torch.std(raw_data)
        raw_data = (raw_data - torch.min(raw_data)) / (torch.max(raw_data) - torch.min(raw_data))
        ## Put things between -1 and 1
        raw_data = (raw_data - 0.5) / 0.5

        # dataset = raw_data[0:1, :traj_length].cpu().numpy()
        # n_envs, n_timesteps, n_dimensions = dataset.shape

        ## Tile the data into -1, traj_length, n_dimensions
        _, raw_timesteps, _ = raw_data.shape
        dataset = []
        for i in range(0, raw_timesteps, traj_length):
            dataset.append(raw_data[0:1, i:i+traj_length, :])
        dataset = np.concatenate(dataset, axis=0)
        n_envs, n_timesteps, n_dimensions = dataset.shape

        t_eval = np.linspace(0, 1., n_timesteps)
        # t_eval = raw_t_eval[:traj_length]
        labels = np.arange(n_envs)

        self.total_envs = n_envs
        self.nb_classes = 64
        self.num_steps = n_timesteps
        self.data_size = n_dimensions

        super().__init__(dataset, labels, t_eval, traj_prop=1.0)



# class DynamicsDataset(TimeSeriesDataset):
#     """
#     For the synthetic control dataset from Time Series Classification
#     """

#     def __init__(self, data_dir, traj_length=1000):
#         try:
#             raw = np.load(data_dir)
#             raw_data = raw["X"][:, :, :traj_length, :]
#             raw_t_eval = raw["t"][:traj_length]
#         except:
#             raise ValueError(f"Data not loadable at {data_dir}")

#         ## Normalise the dataset between 0 and 1
#         # raw_data = (raw_data - torch.mean(raw_data)) / torch.std(raw_data)
#         raw_data = (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))
#         ## Put things between -1 and 1
#         raw_data = (raw_data - 0.5) / 0.5

#         # dataset = raw_data[0, :]
#         dataset = raw_data.reshape(-1, traj_length, raw_data.shape[-1])
#         n_envs, n_timesteps, n_dimensions = dataset.shape

#         # t_eval = raw_t_eval
#         t_eval = np.linspace(0., 1., n_timesteps)
#         labels = np.arange(n_envs)

#         self.total_envs = n_envs
#         self.nb_classes = 64
#         self.num_steps = n_timesteps
#         self.data_size = n_dimensions

#         super().__init__(dataset, labels, t_eval, traj_prop=1.0)






class MNISTDataset(TimeSeriesDataset):
    """
    For the MNIST dataset, where the time series is the pixels of the image, and the example of Trends dataset above
    """

    def __init__(self, data_dir, data_split, mini_res=4, traj_prop=1.0, unit_normalise=False, fashion=False):
        self.nb_classes = 10
        self.num_steps = (28//mini_res)**2
        self.data_size = 1
        self.mini_res = mini_res

        self.traj_prop = traj_prop

        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5) if not unit_normalise else transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x[:, ::mini_res, ::mini_res]) if mini_res>1 else transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x.reshape(self.data_size, self.num_steps).t()),
            ]
        )

        if fashion:
            data = torchvision.datasets.FashionMNIST(
                data_dir, train=True if data_split=="train" else False, download=True, transform=tf
            )
        else:
            data = torchvision.datasets.MNIST(
                data_dir, train=True if data_split=="train" else False, download=True, transform=tf
            )

        ## Get all the data in one large batch (to apply the transform)
        dataset, labels = next(iter(torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)))
        # dataset, labels = next(iter(torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)))

        t_eval = np.linspace(0., 1., self.num_steps)
        self.total_envs = dataset.shape[0]

        super().__init__(dataset.numpy(), labels.numpy(), t_eval, traj_prop=traj_prop)

        # ## Limit the dataset to N samples (For debugging) TODO: Remove this
        # N = 1024
        # self.total_envs = N
        # self.dataset = self.dataset[:N]
        # self.labels = self.labels[:N]





class CIFARDataset(TimeSeriesDataset):
    """
    For the MNIST dataset, where the time series is the pixels of the image, and the example of Trends dataset above
    """

    def __init__(self, data_dir, data_split, mini_res=4, traj_prop=1.0, unit_normalise=False):
        self.nb_classes = 10
        self.num_steps = (32//mini_res)**2
        self.data_size = 3
        self.mini_res = mini_res

        self.traj_prop = traj_prop

        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5) if not unit_normalise else transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x[:, ::mini_res, ::mini_res]) if mini_res>1 else transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x.reshape(self.data_size, self.num_steps).t()),
            ]
        )

        data = torchvision.datasets.CIFAR10(
            data_dir, train=True if data_split=="train" else False, download=True, transform=tf
        )

        ## Get all the data in one large batch (to apply the transform)
        dataset, labels = next(iter(torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)))
        # dataset, labels = next(iter(torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)))

        ## Filter and return cats only: class 3
        dataset = dataset[labels==3]
        labels = labels[labels==3]

        t_eval = np.linspace(0., 1., self.num_steps)
        self.total_envs = dataset.shape[0]

        super().__init__(dataset.numpy(), labels.numpy(), t_eval, traj_prop=traj_prop)










import pandas as pd
from typing import Tuple
from PIL import Image
import os

## Set the seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class CelebADataset(torch.utils.data.Dataset):
    """
    A celeb a dataloader for meta-learning.
    """
    def __init__(self, 
                 data_path="./data/", 
                 data_split="train",
                 num_shots=100,
                 resolution=(32, 32),
                 order_pixels=False,
                 unit_normalise=False):

        if num_shots <= 0:
            raise ValueError("Number of shots must be greater than 0.")
        elif num_shots > resolution[0]*resolution[1]:
            raise ValueError("Number of shots must be less than the total number of pixels.")
        self.nb_shots = num_shots

        self.unit_normalise = unit_normalise
        self.input_dim = 2
        self.output_dim = 3
        self.img_size = (*resolution, self.output_dim)
        self.order_pixels = order_pixels
        ## Read the partitioning file: train(0), val(1), test(2)

        self.data_path = data_path
        partitions = pd.read_csv(self.data_path+'list_eval_partition.txt', 
                                 header=None, 
                                 sep=r'\s+', 
                                 names=['filename', 'partition'])
        if data_split in ["train"]:
            self.files = partitions[partitions['partition'] == 0]['filename'].values
        elif data_split in ["val"]:
            self.files = partitions[partitions['partition'] == 1]['filename'].values
        elif data_split in ["test"]:
            # self.files = partitions[partitions['partition'] == 2]['filename'].values

            ## To get the translation-equivariance img in front of the test set (incl. Ellen selfie)
            self.files = partitions[(partitions['partition'] == 2) | (partitions['partition'] == 3)]['filename'].values
            self.files = np.concatenate((self.files[-1:], self.files[:-1]))

        else:
            raise ValueError(f"Invalid data split provided. Got {data_split}")

        if data_split in ["train", "val"]:
            self.adaptation = False
        elif data_split in ["test"]:
            self.adaptation = True
        else:
            raise ValueError(f"Invalid data split provided. Got {data_split}")

        ## A list of MVPs images (or the worst during self-modulation) - Useful for active learning
        # self.mvp_files = self.files

        self.total_envs = len(self.files)
        if self.total_envs == 0:
            raise ValueError("No files found for the split.")

        self.total_pixels = self.img_size[0] * self.img_size[1]

        ## Ssee CAVIA code: https://github.com/lmzintgraf/cavia)
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                            transforms.Resize((self.img_size[0], self.img_size[1]), Image.LANCZOS),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(mean=0.5, std=0.5) if not False else transforms.Lambda(lambda x: x),
                                            ])

        ## Add everything a time series dataset would have
        self.num_steps = self.total_pixels
        self.data_size = self.output_dim
        self.t_eval = np.linspace(0., 1., self.num_steps)

        self.nb_classes = 40                                        ### If using the attributes
        ## labels as NaNs
        self.labels = np.nan * np.ones((self.total_envs,), dtype=int)


    def get_image(self, filename) -> torch.Tensor:
        img_path = os.path.join(self.data_path+"img_align_celeba/", filename)
        img = self.transform(img_path).float()
        img = img.permute(1, 2, 0)
        return np.array(img)

    def sample_pixels(self, img) -> Tuple[np.ndarray, np.ndarray]:        ## TODO: Stay in torch throughout this function!
        total_pixels = self.img_size[0] * self.img_size[1]

        if self.order_pixels:
            flattened_indices = np.arange(self.nb_shots)
        else:
            flattened_indices = np.random.choice(total_pixels, size=self.nb_shots, replace=False)

        x, y = np.unravel_index(flattened_indices, (self.img_size[0], self.img_size[1]))
        coords = np.vstack((x, y)).T
        normed_coords = (coords / np.array(self.img_size[:2]))

        pixel_values = img[coords[:, 0], coords[:, 1], :]

        return normed_coords, pixel_values

    def set_seed_sample_pixels(self, seed, idx):
        np.random.seed(seed)
        # np.random.set_state(seed)
        img = self.get_image(self.files[idx])
        return self.sample_pixels(img)


    def __getitem__(self, idx):
        img = self.get_image(self.files[idx])
        normed_coords, pixel_values = self.sample_pixels(img)
        pixels = pixel_values.reshape(-1, self.output_dim)

        if not self.unit_normalise:
        ## Rescale the RGB pixels to be between -1 and 1
            pixels = (pixels - 0.5) / 0.5

        return (pixels, self.t_eval), self.labels[idx]


    def __len__(self):
        return self.total_envs









# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     ## Reset numpy random seed
#     np.random.seed(0)

#     data_folder, batch_size = "data/", 128
#     resolution = (16, 16)
#     trainloader = NumpyLoader(CelebADataset(data_folder+"celeba/", data_split="train", num_shots=np.prod(resolution), resolution=resolution, order_pixels=True, unit_normalise=False), 
#                               batch_size=batch_size, 
#                               shuffle=True, 
#                               num_workers=24)

#     batch = next(iter(trainloader))
#     (images, times), labels = batch
#     print("Images shape:", images.shape)
#     print("Labels shape:", labels.shape)

#     print("Min and Max in the dataset:", np.min(images), np.max(images), flush=True)

#     ## Plot a few samples, along with their labels as title in a 4x4 grid (chose them at random)
#     fig, axs = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
#     colors = ['r', 'g', 'b', 'c', 'm', 'y']

#     dataset = "celeba"
#     data_size = 3
#     mini_res_mnist = 4
#     image_datasets = ["mnist", "mnist_fashion", "cifar", "celeba"]
#     def get_width(dataset):
#         if dataset in ["mnist", "mnist_fashion"]:
#             return 28 // mini_res_mnist
#         elif dataset=="cifar":
#             return 32 // mini_res_mnist
#         elif dataset=="celeba":
#             return resolution[0]
#         else:
#             return 32

#     width = get_width(dataset)
#     res = (width, width, data_size)
#     for i in range(4):
#         for j in range(4):
#             idx = np.random.randint(0, images.shape[0])
#             # axs[i, j].imshow(images[idx].reshape(res), cmap='gray', vmin=-1, vmax=1)

#             to_plot = (images[idx].reshape(res) + 1 ) / 2
#             axs[i, j].imshow(to_plot, cmap='gray')

#             axs[i, j].set_title(f"Class: {labels[idx]}", fontsize=12)
#             axs[i, j].axis('off')
































# class MovingMNISTDataset(torch.utils.data.Dataset):
#     """
#     MovingMNIST dataset, returning elements of shape (T, C, H, W)
#     """

#     def __init__(self, data_dir, data_split, mini_res=4, unit_normalise=False):
#         self.nb_classes = 10
#         self.num_steps = 19
#         self.data_size = (1, 64//mini_res, 64//mini_res)
#         self.mini_res = mini_res

#         tf = transforms.Compose(
#             [
#                 transforms.Lambda(lambda x: (x.float() / 255.) * 2 - 1) if not unit_normalise else transforms.Lambda(lambda x: x),
#                 transforms.Lambda(lambda x: x[:, :, ::mini_res, ::mini_res]) if mini_res>1 else transforms.Lambda(lambda x: x),
#             ]
#         )
#         data = torchvision.datasets.MovingMNIST(
#             data_dir, split=data_split, download=True, transform=tf, split_ratio=19 if data_split=="train" else 1,
#         )

#         ## Get all the data in one large batch (to apply the transform)
#         self.dataset = next(iter(torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)))
#         self.labels = np.random.randint(0, 10, size=(self.dataset.shape[0],)) * np.nan

#         self.t_eval = np.linspace(0., 1., self.num_steps)
#         self.total_envs = self.dataset.shape[0]

#     def __getitem__(self, idx):
#         inputs = self.dataset[idx, ...]
#         outputs = self.labels[idx]
#         t_eval = self.t_eval

#         return (inputs, t_eval), outputs

#     def __len__(self):
#         return self.total_envs







class MovingMNISTDataset(torch.utils.data.Dataset):
    """
    MovingMNIST dataset, returning elements of shape (T, C, H, W)
    """

    def __init__(self, data_dir, data_split, mini_res=4, unit_normalise=False):
        self.nb_classes = 10
        self.num_steps = 19
        self.data_size = (1, 64//mini_res, 64//mini_res)
        self.mini_res = mini_res

        tf = transforms.Compose(
            [
                transforms.Lambda(lambda x: (x.float() / 255.) * 2 - 1) if not unit_normalise else transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x[:, :, ::mini_res, ::mini_res]) if mini_res>1 else transforms.Lambda(lambda x: x),
            ]
        )
        self.dataset = torchvision.datasets.MovingMNIST(
            data_dir, split=data_split, download=True, transform=tf, split_ratio=19 if data_split=="train" else 1,
        )

        self.total_envs = len(self.dataset)
        self.labels = np.random.randint(0, 10, size=(self.total_envs,)) * np.nan
        self.t_eval = np.linspace(0., 1., self.num_steps)

    def __getitem__(self, idx):
        inputs = self.dataset[idx]
        outputs = self.labels[idx]
        t_eval = self.t_eval

        return (inputs, t_eval), outputs

    def __len__(self):
        return self.total_envs





if __name__ == "__main__":
    ### Test the MovingMNIST dataset
    import matplotlib.pyplot as plt
    ## Reset numpy random seed
    np.random.seed(0)

    data_folder, batch_size = "data/", 1
    trainloader = NumpyLoader(MovingMNISTDataset(data_folder, data_split="train", mini_res=1, unit_normalise=False), 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=0)
    
    batch = next(iter(trainloader))
    (images, times), labels = batch
    print("Images shape:", images.shape)

    print("Min and Max in the dataset:", np.min(images), np.max(images), flush=True)

    ## Plot the single video in the batch
    video = (images[0] + 1)/2       ## Shape: (T, C, H, W)
    # video = (((images[0] + 1)/2) * 255).astype(int)       ## Shape: (T, C, H, W)
    print("Min an Max in the video:", np.min(video), np.max(video))
    T, C, H, W = video.shape
    nb_frames = video.shape[0]
    fig, axs = plt.subplots(1, T, figsize=(4*T, 4), sharex=True)
    for i in range(T):
        axs[i].imshow(video[i, 0, :, :], cmap='gray')
        axs[i].axis('off')

    plt.show()
    print("Labels:", labels)






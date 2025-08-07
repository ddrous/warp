#%%
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

import pandas as pd
from typing import Tuple
from PIL import Image
import os
import jax.tree as jtree








############################# DATASETS FOR TYPICAL TIME SERIES TASKS #############################

class TimeSeriesDataset:
    """
    Base class for any time series dataset, from which the others will inherit
    """
    def __init__(self, dataset, labels, t_eval, traj_prop=1.0, positional_enc=None):

        self.dataset = dataset
        n_envs, n_timesteps, n_dimensions = dataset.shape

        # self.t_eval = t_eval
        # Let's replace t_eval of shape (n_envs, n_timesteps, 1) with positional encoding of dimention D
        if positional_enc is not None:
            D, PE_cte = positional_enc
            pos_enc = np.zeros((n_timesteps, D))
            for pos in range(n_timesteps):
                for i in range(0, D, 2):
                    pos_enc[pos, i] = np.sin(pos / (PE_cte ** (i / D)))
                    if i + 1 < D:
                        pos_enc[pos, i + 1] = np.cos(pos / (PE_cte ** (i / D)))
            self.t_eval = np.concatenate((t_eval[:, None], pos_enc), axis=-1)
        else:
            self.t_eval = t_eval[:, None]

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
            ## Straightforward approach, no subsampling ###
            return (inputs, t_eval), outputs
        else:
            ## Select a random subset of traj_len-2 indices, then concatenate the start and end points
            indices = np.sort(np.random.choice(np.arange(1,self.num_steps-1), traj_len-2, replace=False))
            indices = np.concatenate(([0], indices, [self.num_steps-1]))
            ts = t_eval[indices]
            trajs = inputs[indices, :]
            return (trajs, ts), outputs

    def __len__(self):
        return self.total_envs


class DynamicsDataset(TimeSeriesDataset):
    """
    For dynamical systems datasets: MSD, Lorentz, etc.
    """

    def __init__(self, data_dir, traj_length=None, normalize=True, min_max=None):
        """
        Args:
            data_dir (str): Path to the data file.
            traj_length (int): Length of the sub-trajectores (long trajectories will be split in chunks before learning). Not used here.
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            min_max (tuple, optional): Tuple of (min, max) values for normalization. Defaults to None.
        """
        try:
            raw_data = np.load(data_dir)
        except:
            raise ValueError(f"Data not loadable at {data_dir}")

        if min_max is not None:
            self.min_data = min_max[0]
            self.max_data = min_max[1]
        else:
            try:
                self.min_data = torch.min(raw_data)
                self.max_data = torch.max(raw_data)
            except:
                self.min_data = np.min(raw_data)
                self.max_data = np.max(raw_data)

        ## Normalise the dataset between 0 and 1, then put it between -1 and 1
        if normalize:
            raw_data = (raw_data - self.min_data) / (self.max_data - self.min_data)
            raw_data = (raw_data - 0.5) / 0.5

        if traj_length is not None and traj_length > 0 and traj_length < raw_data.shape[1]:
            # Tile the data into -1, traj_length, n_dimensions
            _, raw_timesteps, _ = raw_data.shape
            dataset = []
            for e in range(raw_data.shape[0]):
                for i in range(0, raw_timesteps, traj_length):
                    dataset.append(raw_data[e:e+1, i:i+traj_length, :])
            dataset = np.concatenate(dataset, axis=0)
            n_envs, n_timesteps, n_dimensions = dataset.shape
        else:
            dataset = raw_data

        ## One environment is a (new) sample
        n_envs, n_timesteps, n_dimensions = dataset.shape

        t_eval = np.linspace(0., 1., n_timesteps)
        labels = np.arange(n_envs)

        self.total_envs = n_envs
        self.nb_classes = n_envs
        self.num_steps = n_timesteps
        self.data_size = n_dimensions

        super().__init__(dataset, labels, t_eval, traj_prop=1.0)



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
        t_eval = np.linspace(0., 1., n_timesteps)

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



class MNISTDataset(TimeSeriesDataset):
    """
    For the MNIST dataset, where the time series is the flattened image, pixel-by-pixel
    """

    def __init__(self, data_dir, data_split, mini_res=4, traj_prop=1.0, unit_normalise=False, fashion=False, positional_enc=None):
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

        t_eval = np.linspace(0., 1., self.num_steps)
        # ## t_eval is shape: (n_timesteps, 1). Append the normalised x,y pixel coordinates based on the image size, so that t_eval is shape: (n_timesteps, 3)
        # pixel_coords = np.zeros((self.num_steps, 2))
        # for i in range(self.num_steps):
        #     x = (i % (28//mini_res)) / (28//mini_res - 1)
        #     y = (i // (28//mini_res)) / (28//mini_res - 1)
        #     pixel_coords[i, 0] = x
        #     pixel_coords[i, 1] = y
        # t_eval = np.concatenate((t_eval[:, None], pixel_coords), axis=-1)

        self.total_envs = dataset.shape[0]

        super().__init__(dataset.numpy(), labels.numpy(), t_eval, traj_prop=traj_prop, positional_enc=positional_enc)



class CIFARDataset(TimeSeriesDataset):
    """
    For the CIFAR-10 dataset
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

        t_eval = np.linspace(0., 1., self.num_steps)
        self.total_envs = dataset.shape[0]

        super().__init__(dataset.numpy(), labels.numpy(), t_eval, traj_prop=traj_prop)



class SpiralsDataset(TimeSeriesDataset):
    """
    For a spirals dataset: https://docs.kidger.site/diffrax/examples/neural_cde/
    """

    def __init__(self, data_dir, normalize=True, min_max=None):
        try:
            raw_data = np.load(data_dir)
        except:
            raise ValueError(f"Data not loadable at {data_dir}")

        ## Normalise the dataset between 0 and 1
        if normalize:
            raw_data = (raw_data - self.min_data) / (self.max_data - self.min_data)

        dataset = raw_data["xs"]
        n_envs, n_timesteps, n_dimensions = dataset.shape

        t_eval = np.linspace(0., 1., n_timesteps)
        labels = raw_data["ys"].astype(int).squeeze()

        self.total_envs = n_envs
        self.nb_classes = 2
        self.num_steps = n_timesteps
        self.data_size = n_dimensions

        super().__init__(dataset, labels, t_eval, traj_prop=1.0)


class UEADataset(TimeSeriesDataset):
    """
    For a UEA dataset, with preprocessing done by https://github.com/Benjamin-Walker/log-neural-cdes
    """

    def __init__(self, data_dir, normalize=True, min_max=None, positional_enc=None):
        try:
            raw_data = np.load(data_dir)
        except:
            raise ValueError(f"Data not loadable at {data_dir}")

        ## Normalise the dataset between 0 and 1
        dataset = raw_data["data"].astype(np.float32)

        if min_max is not None:
            self.min_data = min_max[0]
            self.max_data = min_max[1]
        else:
            self.min_data = np.min(dataset, axis=(0, 1), keepdims=True)
            self.max_data = np.max(dataset, axis=(0, 1), keepdims=True)

        if normalize:
            dataset = (dataset - self.min_data) / (self.max_data - self.min_data)
            ## Put things between -1 and 1
            dataset = (dataset - 0.5) / 0.5

        n_envs, n_timesteps, n_dimensions = dataset.shape

        t_eval = np.linspace(0., 1., n_timesteps)
        labels = raw_data["labels"].astype(np.int32).squeeze()

        self.total_envs = n_envs
        self.nb_classes = int(np.max(labels)) + 1
        self.num_steps = n_timesteps
        self.data_size = n_dimensions

        super().__init__(dataset, labels, t_eval, traj_prop=1.0, positional_enc=positional_enc)


class PathFinderDataset(UEADataset):
    """
    Alias for the UEADataset (useful  for clarity)
    """
    def __init__(self, data_dir, normalize=True, min_max=None):
        super().__init__(data_dir, normalize=normalize, min_max=min_max)









############################# DATASETS FOR REPEAT-COPY TIME SERIES TASKS #############################

class TimeSeriesRepeatDataset:
    """
    Base class for any time series repeat-copy dataset, from which several others will inherit
    """
    def __init__(self, in_dataset, out_dataset, t_eval, traj_prop=1.0):

        self.in_dataset = in_dataset            ## Input sequences
        self.out_datasets = out_dataset         ## Output sequences
        n_envs, n_timesteps, n_dimensions = in_dataset.shape
        self.t_eval = t_eval
        self.total_envs = n_envs

        if traj_prop < 0 or traj_prop > 1:
            raise ValueError("The smallest proportion of the trajectory to use must be between 0 and 1")
        self.traj_prop = traj_prop
        self.traj_len = int(n_timesteps * traj_prop)

        self.num_steps = n_timesteps
        self.data_size = n_dimensions

    def __getitem__(self, idx):
        inputs = self.in_dataset[idx, :, :]
        outputs = self.out_datasets[idx, :, :]
        t_eval = self.t_eval
        traj_len = self.traj_len

        if self.traj_prop == 1.0:
            ### STRAIGHFORWARD APPROACH ###
            return (inputs, t_eval), outputs
        else:
            ## Select a random subset of traj_len-2 indices, then concatenate the start and end points
            indices = np.sort(np.random.choice(np.arange(1,self.num_steps-1), traj_len-2, replace=False))
            indices = np.concatenate(([0], indices, [self.num_steps-1]))
            ts = t_eval[indices]
            in_trajs = inputs[indices, :]
            out_trajs = outputs[indices, :]
            return (in_trajs, ts), out_trajs

    def __len__(self):
        return self.total_envs


class DynamicsRepeatDataset(TimeSeriesRepeatDataset):
    """
    For the a dynamical system dataset for repeat-copy tasks (e.g. Lotka-Volterra)
    """

    def __init__(self, data_dir, traj_length, min_max=None):
        try:
            raw_data = np.load(data_dir)
            in_raw_data = raw_data["clipped"]
            out_raw_data = raw_data["full"]
        except:
            raise ValueError(f"Data not loadable at {data_dir}")

        ## Normalise the dataset between 0 and 1
        if min_max is not None:
            self.min_data = min_max[0]
            self.max_data = min_max[1]
        else:
            self.min_data = np.min(out_raw_data)
            self.max_data = np.max(out_raw_data)

        in_raw_data = (in_raw_data - self.min_data) / (self.max_data - self.min_data)
        out_raw_data = (out_raw_data - self.min_data) / (self.max_data - self.min_data)

        ## Put things between -1 and 1
        in_raw_data = (in_raw_data - 0.5) / 0.5
        out_raw_data = (out_raw_data - 0.5) / 0.5

        ## Replace the smallest value in in_raw_data with exactly -1 ?
        # in_raw_data[in_raw_data == np.min(in_raw_data)] = -1

        n_envs, n_timesteps, n_dimensions = in_raw_data.shape

        # t_eval = raw_t_eval[:traj_length]
        t_eval = np.linspace(0, 1., n_timesteps)

        self.total_envs = n_envs
        self.nb_classes = n_envs
        self.num_steps = n_timesteps
        self.data_size = n_dimensions

        super().__init__(in_raw_data, out_raw_data, t_eval, traj_prop=1.0)



class ARC_AGIDataset(TimeSeriesRepeatDataset):
    """
    For the a ARC-AGI dataset framed as a repeat-copy tasks
    """

    def __init__(self, data_dir, traj_length, min_max=None):
        # try:
        #     in_raw_data = np.load(data_dir+"in.npy")
        #     out_raw_data = np.load(data_dir+"out.npy")
        # except:
        #     raise ValueError(f"Data not loadable at root {data_dir}")
        in_raw_data = np.load(data_dir+"in.npy")[..., None]
        out_raw_data = np.load(data_dir+"out.npy")[..., None]

        n_envs, n_timesteps, n_dimensions = in_raw_data.shape

        t_eval = np.linspace(0, 1., n_timesteps)

        self.total_envs = n_envs
        self.nb_classes = n_envs
        self.num_steps = n_timesteps
        self.data_size = n_dimensions

        super().__init__(in_raw_data, out_raw_data, t_eval, traj_prop=1.0)









############## SPECIAL DATASETS NOT INHERITING FROM THE TIME SERIES CLASSES ##############

class CelebADataset(torch.utils.data.Dataset):
    """
    For the CelebA dataset, adapted for time series from https://github.com/ddrous/self-mod
    """
    def __init__(self, 
                 data_path="./data/", 
                 data_split="train",
                 num_shots=100,
                 resolution=(32, 32),
                 order_pixels=False,
                 unit_normalise=False,
                 positional_enc=None):

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
        self.positional_enc = positional_enc
        self.t_eval = np.linspace(0., 1., self.num_steps)
        if self.positional_enc is not None:
            D, PE_cte = self.positional_enc
            pos_enc = np.zeros((self.num_steps, D))
            for pos in range(self.num_steps):
                for i in range(0, D, 2):
                    pos_enc[pos, i] = np.sin(pos / (PE_cte ** (i / D)))
                    if i + 1 < D:
                        pos_enc[pos, i + 1] = np.cos(pos / (PE_cte ** (i / D)))
            self.t_eval = np.concatenate((self.t_eval[:, None], pos_enc), axis=-1)
        else:
            self.t_eval = self.t_eval[:, None]

        self.nb_classes = 40                                        ### If using the attributes
        ## labels as NaNs
        self.labels = np.nan * np.ones((self.total_envs,), dtype=int)

    def get_image(self, filename) -> torch.Tensor:
        img_path = os.path.join(self.data_path+"img_align_celeba/", filename)
        img = self.transform(img_path).float()
        img = img.permute(1, 2, 0)
        return np.array(img)

    def sample_pixels(self, img) -> Tuple[np.ndarray, np.ndarray]:
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









class ICLDataset(torch.utils.data.Dataset):
    """
    For the ICL dataset, inspired by https://arxiv.org/abs/2501.16265
    """
    def __init__(self, x_dim=2, seq_len=36, num_envs=128, positional_enc=None):
        self.x_dim = x_dim      ## D
        self.seq_len = seq_len  ## N
        # Define 
        # self.data_gen_matrix = np.random.rand(self.x_dim, self.x_dim).astype(np.float32)        ## data generation matrix \Alpha
        self.data_matrix = np.eye(self.x_dim)
        self.output_matrix = np.eye(self.x_dim)                                                   ## One for each sequence

        self.data_size = x_dim
        self.total_envs = num_envs  ## Number of environments

        self.t_eval = np.linspace(0., 1., seq_len)[:, None]  ## Shape: (N, 1)
        # self.t_eval = np.linspace(0., 1., seq_len*(x_dim+1))[:, None]  ## Shape: (N, 1)

        if positional_enc is not None:
            D, PE_cte = positional_enc
            pos_enc = np.zeros((self.seq_len, D))
            for pos in range(self.seq_len):
                for i in range(0, D, 2):
                    pos_enc[pos, i] = np.sin(pos / (PE_cte ** (i / D)))
                    if i + 1 < D:
                        pos_enc[pos, i + 1] = np.cos(pos / (PE_cte ** (i / D)))
            self.t_eval = np.concatenate((self.t_eval, pos_enc), axis=-1)

    def __len__(self):
        return self.total_envs

    def __getitem__(self, idx):
        """
        Returns a tuple of (inputs, t_eval), outputs
        """
        ## Generate a sequence of shapce (seq_len, x_dim) from the normal distribution with mean 0 and covariance the data generation matrix
        Xs = np.random.multivariate_normal(mean=np.zeros(self.x_dim), cov=self.data_matrix, size=self.seq_len).astype(np.float32)

        ## Sample a random output matrix for this sequence
        output_matrix = np.random.multivariate_normal(mean=np.zeros(self.x_dim), cov=self.output_matrix, size=1).astype(np.float32)[0]

        ## Outputs are generated by multiplying the inputs by the output matrix, independently for each sequence
        ys = Xs@output_matrix  ## Shape: (seq_len, x_dim)
        # print("Shapes before multiplication:", output_matrix.shape, inputs.shape, outputs.shape, flush=True)

        ### Concatenat Xs to Xs (but shifted so that each x containts the next x. The last x's concatenation is set to 0) ###
        # next_Xs = np.zeros_like(Xs)
        # next_Xs[:-1, :] = Xs[1:, :]  ## Shift the inputs by one time step
        # Xs = np.concatenate((Xs, next_Xs), axis=-1)  ## Shape: (seq_len, x_dim + x_dim)
        ######=======================================######

        # ##### Trick to get rig of the randomness in the Xs. At each position, we append the next terms to the sequence in a new row. The resulting sequence is a lower triangular matrix #####
        # next_Xs = np.zeros((self.seq_len, self.seq_len))
        # next_Xs[0, :] = Xs[:, 0]  ## The first row
        # for i in range(1, self.seq_len):
        #     next_Xs[i, :-i] = Xs[i:, 0]  ## Shift the inputs by i time step (1-d case)
        # # print("These are the two values of interest:\n", next_Xs[:, :], Xs[-1, 0], flush=True)
        # # assert next_Xs[-1, 0] == Xs[-1, 0], "Asserting that the entries match"  ## The last input is the same as the last output
        # # Xs = np.concatenate((Xs, next_Xs), axis=-1)  ## Shape: (seq_len, x_dim + 1)
        # Xs = next_Xs
        # ######=======================================######

        ## Another trick, we treat the Xs as a vector of (seq_len, 1), and we repeat is to have a shape of (seq_len, seq_len)
        Xs = np.repeat(Xs, self.seq_len, axis=1)  ## Shape: (seq_len, seq_len)
        # print("Xs:", Xs, flush=True)
        # exit(0)

        ## Concatenate the inputs and outputs to have a shape of (seq_len, x_dim + 1)
        outputs = np.concatenate((Xs, ys[:, None]), axis=-1)    # Shape: (seq_len, x_dim + 1)

        # ## Flattent the sequence into (seq_len* (x_dim + 1),)
        # outputs = outputs.reshape(-1, 1)

        ## Set the last input to 0
        inputs = outputs.copy()
        inputs[-1, -1] = 0

        # return (sequence, self.t_eval), np.NaN
        # return (inputs, self.t_eval), in_out_seq

        return (inputs, self.t_eval), ys[:, None]
        # return (inputs, self.t_eval), outputs
        # return (outputs, self.t_eval), np.array([0])[0]





################################ PUTTING IT ALL TOGETHER ########################################

def numpy_collate(batch):
  return jtree.map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

    self.num_batches = np.ceil(len(dataset) / batch_size).astype(int)



def make_dataloaders(data_folder, config):
    """
    Create the dataset and dataloaders for the given dataset
    """

    ## Get the parameters
    dataset = config["general"]["dataset"]
    batch_size = config["training"]["batch_size"]
    positional_enc = config["data"].get("positional_encoding", None)

    if dataset in ["mnist", "mnist_fashion"]:
        # ### MNIST Classification (From Sacha Rush's Annotated S4)
        print(" #### MNIST Dataset ####")
        fashion = dataset=="mnist_fashion"
        downsample_factor = config["data"]["downsample_factor"]

        trainloader = NumpyLoader(MNISTDataset(data_folder, data_split="train", mini_res=downsample_factor, traj_prop=1.0, unit_normalise=False, fashion=fashion, positional_enc=positional_enc), 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=24)
        testloader = NumpyLoader(MNISTDataset(data_folder, data_split="test", mini_res=downsample_factor, traj_prop=1.0, unit_normalise=False, fashion=fashion, positional_enc=positional_enc),
                                    batch_size=batch_size,
                                    shuffle=False, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        min_res = 28 // downsample_factor

    elif dataset=="cifar":
        print(" #### CIFAR Dataset ####")
        downsample_factor = config["data"]["downsample_factor"]

        trainloader = NumpyLoader(CIFARDataset(data_folder, data_split="train", mini_res=downsample_factor, traj_prop=1.0, unit_normalise=False), 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        testloader = NumpyLoader(CIFARDataset(data_folder, data_split="test", mini_res=downsample_factor, traj_prop=1.0, unit_normalise=False),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        min_res = 32 // downsample_factor

    elif dataset=="celeba":
        print(" #### CelebA Dataset ####")
        resolution = config["data"]["resolution"]

        trainloader = NumpyLoader(CelebADataset(data_folder+"celeba/", data_split="train", num_shots=np.prod(resolution), resolution=resolution, order_pixels=True, unit_normalise=False, positional_enc=positional_enc), 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        testloader = NumpyLoader(CelebADataset(data_folder+"celeba/", data_split="test", num_shots=np.prod(resolution), resolution=resolution, order_pixels=True, unit_normalise=False, positional_enc=positional_enc),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        min_res = min(resolution)

    elif dataset in ["lorentz63", "mass_spring_damper", "cheetah", "electricity"]:
        print(" #### Dynamics Dataset ####")
        normalize = config["data"]["normalize"]
        # traj_len = config["data"]["traj_length"]          ## @TODO: eventually make this parameter
        traj_len = None

        trainloader = NumpyLoader(DynamicsDataset(data_folder+"train.npy", traj_length=traj_len, normalize=normalize, min_max=None), 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        min_max = (trainloader.dataset.min_data, trainloader.dataset.max_data)
        testloader = NumpyLoader(DynamicsDataset(data_folder+"test.npy", traj_length=traj_len, normalize=normalize, min_max=min_max),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        print("Training sequence length:", seq_length)
        min_res = None

    elif dataset in ["lotka"]:
        print(" #### Dynamics-Repeat Dataset ####")
        traj_len = None

        trainloader = NumpyLoader(DynamicsRepeatDataset(data_folder+"train.npz", traj_length=traj_len, min_max=None), 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        min_max = (trainloader.dataset.min_data, trainloader.dataset.max_data)
        testloader = NumpyLoader(DynamicsRepeatDataset(data_folder+"test.npz", traj_length=traj_len, min_max=min_max),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        print("Training sequence length:", seq_length)
        min_res = None

    elif dataset in ["arc_agi"]:
        print(" #### ARC-AGI Dataset ####")
        traj_len = None

        trainloader = NumpyLoader(ARC_AGIDataset(data_folder+"train_", traj_length=traj_len, min_max=None), 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        testloader = NumpyLoader(ARC_AGIDataset(data_folder+"test_", traj_length=traj_len, min_max=None),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        print("Training sequence length:", seq_length)
        min_res = None

    elif dataset in ["icl"]:
        print(" #### ICL Dataset ####")
        traj_len = config["data"]["sequence_length"]
        data_size = config["data"]["data_dim"]
        seq_length = config["data"]["sequence_length"]        ## distinguish the query from the rest?
        num_envs = config["training"]["batch_size"]*1

        trainloader = NumpyLoader(ICLDataset(x_dim=data_size, seq_len=seq_length, num_envs=num_envs, positional_enc=positional_enc),
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        testloader = NumpyLoader(ICLDataset(x_dim=data_size, seq_len=seq_length, num_envs=num_envs, positional_enc=positional_enc),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        nb_classes = -1
        # data_size += 1  ## +1 for the output

        # seq_length = seq_length * (data_size+1)
        # data_size = 1
        # data_size = 2*data_size + 1  ## +1 for the output
        data_size = seq_length + 1  ## +1 for the output

        print("Training sequence length:", seq_length)
        min_res = None

    elif dataset in ["spirals"]:
        trainloader = NumpyLoader(SpiralsDataset(data_folder+"train.npz", normalize=False, min_max=None),
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        testloader = NumpyLoader(SpiralsDataset(data_folder+"test.npz", normalize=False, min_max=None),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        print("Training sequence length:", seq_length)
        min_res = None

    elif dataset in ["uea"]:
        normalize = config["data"]["normalize"]

        trainloader = NumpyLoader(UEADataset(data_folder+"train.npz", normalize=normalize, min_max=None, positional_enc=positional_enc),
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        min_max = (trainloader.dataset.min_data, trainloader.dataset.max_data)
        valloader = NumpyLoader(UEADataset(data_folder+"val.npz", normalize=normalize, min_max=min_max, positional_enc=positional_enc),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        testloader = NumpyLoader(UEADataset(data_folder+"test.npz", normalize=normalize, min_max=min_max, positional_enc=positional_enc),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        print("Training sequence length:", seq_length)
        min_res = None

    elif dataset in ["pathfinder"]:
        trainloader = NumpyLoader(PathFinderDataset(data_folder+"train.npz", normalize=False, min_max=None),
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        valloader = NumpyLoader(PathFinderDataset(data_folder+"val.npz", normalize=False, min_max=None),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        testloader = NumpyLoader(PathFinderDataset(data_folder+"test.npz", normalize=False, min_max=None),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        print("Training sequence length:", seq_length)
        min_res = 32

    elif dataset in ["sine"]:
        normalize = config["data"]["normalize"]
        traj_len = None

        trainloader = NumpyLoader(DynamicsDataset(data_folder+"train.npy", traj_length=traj_len, normalize=normalize, min_max=None),
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        min_max = (trainloader.dataset.min_data, trainloader.dataset.max_data)
        valloader = NumpyLoader(DynamicsDataset(data_folder+"val.npy", traj_length=traj_len, normalize=normalize, min_max=min_max),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        testloader = NumpyLoader(DynamicsDataset(data_folder+"test.npy", traj_length=traj_len, normalize=normalize, min_max=min_max),
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        print("Training sequence length:", seq_length)
        min_res = None

    elif dataset in ["trends"]:
        print(" #### Trends (Synthetic Control) Dataset ####")
        trainloader = NumpyLoader(TrendsDataset(data_folder+"trends/", skip_steps=1, traj_prop=1.0), 
                                batch_size=batch_size if batch_size<600 else 600, 
                                shuffle=True)
        testloader = NumpyLoader(TrendsDataset(data_folder+"trends/", skip_steps=1, traj_prop=1.0), 
                                batch_size=batch_size if batch_size<600 else 600,
                                shuffle=False)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size

        min_res = None

    else:
        raise ValueError(f"Dataset {dataset} not recognised. Please check the configuration file.")


    ## Check if valloader is defined
    if "valloader" not in locals():
        valloader = testloader
        print("Validation set same as test set")

    return trainloader, valloader, testloader, (nb_classes, seq_length, data_size, min_res)












################################ TESTING THE FILES ########################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_folder, batch_size = "data/", 1
    trainloader = NumpyLoader(MovingMNISTDataset(data_folder, data_split="train", mini_res=1, unit_normalise=False), 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=0)
    
    batch = next(iter(trainloader))
    (images, times), labels = batch
    print("Images shape:", images.shape)

    print("Min and Max in the moving images:", np.min(images), np.max(images), flush=True)

    ## Plot the single video in the batch
    video = (images[0] + 1)/2       ## Shape: (T, C, H, W)
    print("Min an Max in the rescaled images:", np.min(video), np.max(video))
    T, C, H, W = video.shape
    nb_frames = video.shape[0]
    fig, axs = plt.subplots(1, T, figsize=(4*T, 4), sharex=True)
    for i in range(T):
        axs[i].imshow(video[i, 0, :, :], cmap='gray')
        axs[i].axis('off')

    plt.show()
    print("Labels:", labels)



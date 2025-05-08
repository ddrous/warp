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






class TimeSeriesDataset:
    """
    Base class for any time series dataset, from which the others will inherit
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



class LorentzDataset(TimeSeriesDataset):
    """
    For the Lorentz dataset from https://arxiv.org/abs/2410.04814
    """

    def __init__(self, data_dir, traj_length):
        try:
            raw_data = torch.load(data_dir)
            raw_t_eval = np.linspace(0, 1., raw_data.shape[1])
        except:
            try:
                raw_data = np.load(data_dir)
                raw_t_eval = np.linspace(0, 1., raw_data.shape[1])
            except:
                raise ValueError(f"Data not loadable at {data_dir}")

        ## Normalise the dataset between 0 and 1
        # raw_data = (raw_data - torch.mean(raw_data)) / torch.std(raw_data)
        try:
            raw_data = (raw_data - torch.min(raw_data)) / (torch.max(raw_data) - torch.min(raw_data))
        except:
            raw_data = (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))

            ## Normalise each channel separately (take the min/max along everything by the last axis)
            # raw_data = (raw_data - np.min(raw_data, axis=(0, 1), keepdims=True)) / (np.max(raw_data, axis=(0, 1), keepdims=True) - np.min(raw_data, axis=(0, 1), keepdims=True))

        ## Put things between -1 and 1
        raw_data = (raw_data - 0.5) / 0.5

        ## Tile the data into -1, traj_length, n_dimensions
        _, raw_timesteps, _ = raw_data.shape
        dataset = []
        for e in range(raw_data.shape[0]):
        # for e in range(16):
            for i in range(0, raw_timesteps, traj_length):
                dataset.append(raw_data[e:e+1, i:i+traj_length, :])
        dataset = np.concatenate(dataset, axis=0)
        n_envs, n_timesteps, n_dimensions = dataset.shape

        # t_eval = raw_t_eval[:traj_length]
        t_eval = np.linspace(0, 1., n_timesteps)
        labels = np.arange(n_envs)

        self.total_envs = n_envs
        self.nb_classes = n_envs
        self.num_steps = n_timesteps
        self.data_size = n_dimensions

        super().__init__(dataset, labels, t_eval, traj_prop=1.0)



class DynamicsDataset(TimeSeriesDataset):
    """
    For all other dyanmics dataset, e.g. from https://arxiv.org/abs/2405.02154
    """

    def __init__(self, data_dir, traj_length=1000):
        try:
            raw = np.load(data_dir)
            raw_data = raw["X"][:, :, :traj_length, :]
            raw_t_eval = raw["t"][:traj_length]
        except:
            raise ValueError(f"Data not loadable at {data_dir}")

        ## Normalise the dataset between 0 and 1
        # raw_data = (raw_data - torch.mean(raw_data)) / torch.std(raw_data)
        raw_data = (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))
        ## Put things between -1 and 1
        raw_data = (raw_data - 0.5) / 0.5

        # dataset = raw_data[0, :]
        dataset = raw_data.reshape(-1, traj_length, raw_data.shape[-1])
        n_envs, n_timesteps, n_dimensions = dataset.shape

        t_eval = raw_t_eval
        # t_eval = np.linspace(0., 1., n_timesteps)
        labels = np.arange(n_envs)

        self.total_envs = n_envs
        self.nb_classes = 64
        self.num_steps = n_timesteps
        self.data_size = n_dimensions

        super().__init__(dataset, labels, t_eval, traj_prop=1.0)




class MNISTDataset(TimeSeriesDataset):
    """
    For the MNIST dataset, where the time series is the flattened image, pixel-by-pixel
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
        # dataset, labels = next(iter(torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)))

        ## Filter and return cats only: class 3 (to make the task easier)
        # dataset = dataset[labels==3]
        # labels = labels[labels==3]

        t_eval = np.linspace(0., 1., self.num_steps)
        self.total_envs = dataset.shape[0]

        super().__init__(dataset.numpy(), labels.numpy(), t_eval, traj_prop=traj_prop)






####### SPECIAL CLASSES NOT INHERITING FROM THE BASE TIMESERIES CLASS #######

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

########################################################################









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

    if dataset in ["mnist", "mnist_fashion"]:
        # ### MNIST Classification (From Sacha Rush's Annotated S4)
        print(" #### MNIST Dataset ####")
        fashion = dataset=="mnist_fashion"
        downsample_factor = config["data"]["downsample_factor"]

        trainloader = NumpyLoader(MNISTDataset(data_folder, data_split="train", mini_res=downsample_factor, traj_prop=1.0, unit_normalise=False, fashion=fashion), 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=24)
        testloader = NumpyLoader(MNISTDataset(data_folder, data_split="test", mini_res=downsample_factor, traj_prop=1.0, unit_normalise=False, fashion=fashion),
                                    batch_size=batch_size,
                                    shuffle=True, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        min_res = 28 // downsample_factor

    elif dataset=="cifar":
        print(" #### CIFAR Dataset ####")
        downsample_factor = config["downsample_factor"]

        trainloader = NumpyLoader(CIFARDataset(data_folder, data_split="train", mini_res=downsample_factor, traj_prop=1.0, unit_normalise=False), 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        testloader = NumpyLoader(CIFARDataset(data_folder, data_split="test", mini_res=downsample_factor, traj_prop=1.0, unit_normalise=False),
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        min_res = 32 // downsample_factor

    elif dataset=="celeba":
        print(" #### CelebA Dataset ####")
        resolution = config["data"]["resolution"]

        trainloader = NumpyLoader(CelebADataset(data_folder+"celeba/", data_split="train", num_shots=np.prod(resolution), resolution=resolution, order_pixels=True, unit_normalise=False), 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        testloader = NumpyLoader(CelebADataset(data_folder+"celeba/", data_split="test", num_shots=np.prod(resolution), resolution=resolution, order_pixels=True, unit_normalise=False),
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        min_res = min(resolution)

    elif dataset in ["lorentz63"]:                  ## TODO: Fix this
        print(" #### Dynamics Dataset ####")
        # data_url = "dynamics/lorentz-63/full.pt"
        data_url = "dynamics/mass-spring-damper/train.npy"
        traj_len = 1000

        trainloader = NumpyLoader(LorentzDataset(data_folder+data_url, traj_length=traj_len), 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=24)
        testloader = NumpyLoader(LorentzDataset(data_folder+data_url, traj_length=traj_len),     ## 5500
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=24)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size
        print("Training sequence length:", seq_length)
        min_res = None

    else:
        print(" #### Trends (Synthetic Control) Dataset ####")
        ## ======= below to run the easy Trends dataset instead!
        trainloader = NumpyLoader(TrendsDataset(data_folder+"trends/", skip_steps=1, traj_prop=1.0), 
                                batch_size=batch_size if batch_size<600 else 600, 
                                shuffle=True)
        testloader = NumpyLoader(TrendsDataset(data_folder+"trends/", skip_steps=1, traj_prop=1.0), 
                                batch_size=batch_size if batch_size<600 else 600,
                                shuffle=True)
        nb_classes, seq_length, data_size = trainloader.dataset.nb_classes, trainloader.dataset.num_steps, trainloader.dataset.data_size

        min_res = None

    return trainloader, testloader, (nb_classes, seq_length, data_size, min_res)













if __name__ == "__main__":
    ### Test the MovingMNIST dataset
    import matplotlib.pyplot as plt
    ## Reset numpy random seed
    # np.random.seed(0)

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
    # video = (((images[0] + 1)/2) * 255).astype(int)       ## Shape: (T, C, H, W)
    print("Min an Max in the rescaled images:", np.min(video), np.max(video))
    T, C, H, W = video.shape
    nb_frames = video.shape[0]
    fig, axs = plt.subplots(1, T, figsize=(4*T, 4), sharex=True)
    for i in range(T):
        axs[i].imshow(video[i, 0, :, :], cmap='gray')
        axs[i].axis('off')

    plt.show()
    print("Labels:", labels)



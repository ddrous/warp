# Weight Space Models

# How to use
- Edit the config
- run the train script

Backups are created in a special folder. If inference is required, those files should be run instead.







# Hyperparameters
This section outlines the configuration parameters used for training and running the model.

## General Parameters

| Parameter          | Description                                                    |
|-------------------|----------------------------------------------------------------|
| `seed`            | Random seed for reproducibility.                               |
| `train`           | Whether to train the model or not.                               |
| `dataset`         | Dataset to use (mnist, cifar, trends, mnist_fashion, dynamics). |
| `data_folder`     | Path to the data folder.                                        |
| `supervision_task`| Type of task (classification, reconstruction).                  |

## Model Parameters

| Parameter                    | Description                                                              |
|-----------------------------|--------------------------------------------------------------------------|
| `mlp_hidden_size`           | Size of hidden layers in MLP.                                           |
| `mlp_depth`                 | Number of layers in MLP.                                               |
| `rnn_inner_dims`            | Dimensions of inner RNN layers.                                          |
| `full_matrix_A`             | Whether to use a full matrix A or a diagonal one.                           |
| `use_theta_prev`            | Whether to use the previous theta in computing the next one.                 |
| `weights_lim`               | Limit for the weights of the root model.                                  |
| `mean_tanh_activation`      | Whether to apply tanh to mean activations.                               |
| `std_additional_tanh`       | Whether to apply additional tanh to standard deviation activations.         |
| `include_canonical_coords`  | Whether to include canonical coordinates in root network input.           |

## Optimizer Parameters

| Parameter            | Description                                   |
|----------------------|-----------------------------------------------|
| `init_lr`            | Initial learning rate.                           |
| `gradient_lim`       | Gradient limit for clipping.                          |
| `lr_decrease_factor` | Factor to reduce learning rate on plateau.        |

## Training Parameters

| Parameter                  | Description                                                                |
|---------------------------|----------------------------------------------------------------------------|
| `nb_epochs`                | Number of training epochs.                                                |
| `batch_size`               | Batch size for training.                                                  |
| `print_every`              | How often to print training progress.                                       |
| `unit_normalise`           | Whether to normalize units.                                                |
| `grounding_length`         | Length of grounding pixel for autoregressive digit generation.             |
| `autoregressive_inference` | Type of inference (True for autoregressive, False for memory-based).       |
| `traj_train_prop`          | Proportion of steps to sample for training each time series.             |
| `nb_recons_loss_steps`     | Number of steps to sample for reconstruction loss.                        |
| `train_strategy`           | Training strategy (flip_coin, teacher_forcing, always_true).               |
| `use_mse_loss`             | Whether to use MSE loss.                                                  |
| `forcing_prob`             | Probability for teacher forcing.                                         |
| `std_lower_bound`          | Lower bound for standard deviation.                                      |

## Data Parameters

| Parameter          | Description                          |
|-------------------|--------------------------------------|
| `resolution`      | Resolution of input images.             |
| `mini_res_mnist`  | Mini resolution for MNIST.             |
| `image_datasets`  | List of available image datasets.     |
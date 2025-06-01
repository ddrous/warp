# WARP: Weight-space Adaptive Recurrent Prediction


# How to use
1. Install the requirements: `pip install -r requirements.txt`
2. Edit the config.yaml (see below)
3. Run the main script: `python main.py config.yaml`
4. Navigate to the (newly created) save folder, and analyze the results in `runs/`, e.g. rerun the main script


> Several configuration files corresponding to the experiments presented in the paper can be found in `cfgs/`





# Configuration
This section outlines the configuration parameters used for training and evaluating the models.

## General Parameters

| Parameter          | Description                                                    |
|-------------------|----------------------------------------------------------------|
| `seed`            | Random seed for reproducibility.                               |
| `train`           | Whether to train the model or not.                               |
| `dataset`         | Dataset to use (mnist, celeba, uea, etc.). |
| `data_folder`     | Path to the data folder.                                        |
| `classification`  | Whether this is a classification or a forecasting task.       |


## Data Parameters

| Parameter          | Description                          |
|-------------------|--------------------------------------|
| `resolution`      | Resolution of input images (for CelebA).             |
| `downsample_factor`  | Resolution downsampling factor for MNIST.             |
| `traj_length`  | Unused !     |
| `normalize`  | Whether to place the data in the [-1,1].     |

## Model Parameters

| Parameter                    | Description                                                              |
|-----------------------------|--------------------------------------------------------------------------|
| `model_type`            | 'wsm' for WARP, 'lstm', or 'gru'                 |
| `root_hidden_size`           | Size of hidden layers in the root network.                                           |
| `root_depth`                 | Number of layers in the root network.                                               |
| `root_activation`                 | Activation function for the root MLP.                                               |
| `root_final_activation`                 | Final activation function for the root MLP's output mean.                                               |
| `std_lower_bound`                 | Lower bound for clipping the standard deviation.                                               |
| `nb_rnn_layers`            | Number of RNN layers (fixed at 1 for now !).                                          |
| `init_state_layers`             | Number of layers in the initial hypernetwork (null if sidestepping it completely).                           |
| `input_prev_data`            | Whether the root network uses the previous observation as input.                 |
| `weights_lim`               | Limit for the weights of the root model.                                  |
| `time_as_channel`      | Whether to time as an additional input channel.                               |
| `forcing_prob`       | Probability of using the true input during teacher-forcing.         |
| `noise_theta_init`  | Whether to add noise the theta_0 at the start of the recurrence.           |

## Optimizer Parameters

| Parameter            | Description                                   |
|----------------------|-----------------------------------------------|
| `init_lr`            | Initial learning rate.                           |
| `gradient_lim`       | Gradient limit for clipping.                          |
| `on_plateau` | Set of parameters for the 'reduce_on_pleatau' strategy in Optax.        |


## Training Parameters

| Parameter                  | Description                                                                |
|---------------------------|----------------------------------------------------------------------------|
| `nb_epochs`                | Number of training epochs.                                                |
| `batch_size`               | Batch size for training.                                                  |
| `print_every`              | How often to print training progress.                                       |
| `save_every`           | How often to save artefacts, e.g. losses.                                                |
| `valid_every`           | How often to validate on the validation set                                                |
| `val_criterion`           | Metric for choosing the best model                     |
| `inference_start`         | Length of context length for autoregressive digit generation.             |
| `autoregressive`          | Wether to train in AR mode or convolution mode       |
| `stochastic`          | Wether to use the reparametrization trick, or simply take the mean             |
| `smooth_inference`             | Whether not to use the the reparametrization trick during inference.                                                  |
| `nb_recons_loss_steps`     | Number of steps to sample for reconstruction loss.                        |
| `use_nll_loss`             | Whether to use NLL or MSE loss for forecasting.                                                  |

#%%[markdown]

# ## Generative Recurrent Neural Networks in weight space

#%%

# %load_ext autoreload
# %autoreload 2

from utils import *
from loaders import *
from models import *

import jax
print("\n+=+=+=+=+ Training Weight Space Model +=+=+=+=+")
print("Available devices:", jax.devices())

from jax import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)
# from jax.experimental import checkify

import numpy as np
import jax.numpy as jnp
import equinox as eqx
import torch
import optax

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_theme(context='poster', 
             style='ticks',
             font='sans-serif', 
             font_scale=1, 
             color_codes=True, 
             rc={"lines.linewidth": 1})
mpl.rcParams['savefig.facecolor'] = 'w'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['savefig.bbox'] = 'tight'

from IPython.display import display, Math
import yaml
import argparse
import os
import time
import sys
import pickle

import warnings
warnings.filterwarnings("ignore")








#%%

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

if _in_ipython_session:
    args = argparse.Namespace(config_file='config.yaml')
    print("Notebook session: Using default config.yaml file")
else:
    if len(sys.argv) == 1:
        # Use default config.yaml file
        args = argparse.Namespace(config_file='config.yaml')
        print("No config file provided: Using default config.yaml file")
    elif len(sys.argv) > 1:
        # Use command line argument as config file
        args = argparse.Namespace(config_file=sys.argv[1])
        print(f"Using command line {sys.argv[1]} as config file")
    else:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

with open(args.config_file, 'r') as file:
    config = yaml.safe_load(file)

seed = config['general']['seed']
main_key = jax.random.PRNGKey(seed)
np.random.seed(seed)
torch.manual_seed(seed)






#%%
train = config['general']['train']
data_path = config['general']['data_path']
save_path = config['general']['save_path']

### Create and setup the run and data folders
if train:
    if save_path is not None:
        run_folder = save_path
    else:
        run_folder = make_run_folder('./runs/')
    data_folder = data_path
else:
    run_folder = "./"
    data_folder = f"../../{data_path}"

print("Using run folder:", run_folder)
## Copy the module files to the run folder
logger, checkpoints_folder, plots_folder, artefacts_folder = setup_run_folder(run_folder, training=train)

## Copy the config file to the run folder, renaming it as config.yaml
if not os.path.exists(run_folder+"config.yaml"):
    # os.system(f"cp {args.config_file} {run_folder}config.yaml")
    test_config = config.copy()
    test_config['general']['train'] = False             ## Set the train flag to False
    with open(run_folder+"config.yaml", 'w') as file:
        yaml.dump(test_config, file)

## Print the config file using the logger and pprint
logger.info(f"Config file: {args.config_file}")
logger.info("==== Config file's contents ====")
# pprint(config)

## Log the config using the logger
for key, value in config.items():
    if isinstance(value, dict):
        logger.info(f"{key}:")
        for sub_key, sub_value in value.items():
            logger.info(f"  {sub_key}: {sub_value}")
    else:
        logger.info(f"{key}: {value}")







#%%

trainloader, testloader, data_props = make_dataloaders(data_folder, config)
nb_classes, seq_length, data_size, width = data_props

batch = next(iter(testloader))
(in_sequence, times), output = batch
logger.info(f"Input sequence shape: {in_sequence.shape}")
logger.info(f"Labels/OutputSequence shape: {output.shape}")
logger.info(f"Seq length: {seq_length}")
logger.info(f"Data size: {data_size}")
logger.info(f"Min/Max in the dataset: {np.min(in_sequence), np.max(in_sequence)}")
logger.info("Number of batches:")
logger.info(f"  - Train: {trainloader.num_batches}")
logger.info(f"  - Test: {testloader.num_batches}")

## Plot a few samples in a 4x4 grid (chose them at random)
fig, axs = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
colors = ['r', 'g', 'b', 'c', 'm', 'y']

dataset = config['general']['dataset']
image_datasets = ["mnist", "mnist_fashion", "cifar", "celeba"]
dynamics_datasets = ["lorentz63", "lorentz96", "lotka", "trends", "mass_spring_damper"]
repeat_datasets = ["lotka"]

res = (width, width, data_size)
dim0, dim1 = (0, 1)
for i in range(4):
    for j in range(4):
        idx = np.random.randint(0, in_sequence.shape[0])
        if dataset in image_datasets:
            to_plot = in_sequence[idx].reshape(res)
            if dataset=="celeba":
                to_plot = (to_plot + 1) / 2
            axs[i, j].imshow(to_plot, cmap='gray')
        elif dataset=="trends":
            axs[i, j].plot(in_sequence[idx], color=colors[output[idx]])
        elif dataset in repeat_datasets:
            # axs[i, j].plot(in_sequence[idx, :, dim0], in_sequence[idx, :, dim1], color=colors[(i*j)%len(colors)])
            # axs[i, j].plot(output[idx, :, dim0], output[idx, :, dim1], color=colors[(i*j)%len(colors)], linestyle='--')
            ## Make 4 plots against time steps instead
            axs[i, j].plot(output[idx, :, dim0], color=colors[(i*j)%len(colors)], linestyle='-', lw=1, alpha=0.5)
            axs[i, j].plot(output[idx, :, dim1], color=colors[(i*j)%len(colors)], linestyle='--', lw=1, alpha=0.5)
            axs[i, j].plot(in_sequence[idx, :, dim0], color=colors[(i*j)%len(colors)], lw=3)
            axs[i, j].plot(in_sequence[idx, :, dim1], color=colors[(i*j)%len(colors)], linestyle='--', lw=3)
        else:
            # axs[i, j].plot(in_sequence[idx, :, dim0], in_sequence[idx, :, dim1], color=colors[output[idx]%len(colors)])
            axs[i, j].plot(in_sequence[idx, :, dim0], color=colors[output[idx]%len(colors)], lw=3)
            axs[i, j].plot(in_sequence[idx, :, dim1], color=colors[output[idx]%len(colors)], linestyle='--', lw=3)

        if dataset not in repeat_datasets:
            axs[i, j].set_title(f"Class: {output[idx]}", fontsize=12)
        axs[i, j].axis('off')

plt.suptitle(f"{dataset.upper()} Training Samples", fontsize=20)
plt.draw();
plt.savefig(plots_folder+"samples_train.png", dpi=100, bbox_inches='tight')


# %% Define the model and loss function

model_key, train_key = jax.random.split(main_key, num=2)
model = make_model(model_key, data_size, config)
untrained_model = model

nb_recons_loss_steps = config['training']['nb_recons_loss_steps']
use_nll_loss = config['training']['use_nll_loss']

def loss_fn(model, batch, key):
    """ Loss function for the model. A batch contains: (Xs, Ts), Ys
    Xs: (batch, time, data_size)
    Ts: (batch, time)
    Ys: (batch, num_classes)
    """
    (X_true, times), X_true_out = batch

    X_recons = model(X_true, times, key, inference_start=None)

    if nb_recons_loss_steps is not None:  ## Use all the steps
        ## Randomly sample some steps in the sequence for the loss
        batch_size, nb_timesteps = X_true.shape[0], X_true.shape[1]
        indices_0 = jnp.arange(batch_size)
        indices_1 = jax.random.randint(key, (batch_size, nb_recons_loss_steps), 0, nb_timesteps)

        X_recons_ = jnp.stack([X_recons[indices_0, indices_1[:,j]] for j in range(nb_recons_loss_steps)], axis=1)

        if dataset not in repeat_datasets:
            X_true_ = jnp.stack([X_true[indices_0, indices_1[:,j]] for j in range(nb_recons_loss_steps)], axis=1)
        else:
            X_true_ = jnp.stack([X_true_out[indices_0, indices_1[:,j]] for j in range(nb_recons_loss_steps)], axis=1)

    else:
        X_recons_ = X_recons
        if dataset not in repeat_datasets:
            X_true_ = X_true
        else:
            X_true_ = X_true_out

    if use_nll_loss:
        means = X_recons_[:, :, :data_size]
        stds = X_recons_[:, :, data_size:]
        loss_r = jnp.log(stds) + 0.5*((X_true_ - means)/stds)**2
    else:
        loss_r = optax.l2_loss(X_recons_, X_true_)

    loss = jnp.mean(loss_r)
    return loss, (loss,)


@eqx.filter_jit
def train_step(model, batch, opt_state, key):
    print('\nCompiling function "train_step" ...')

    (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, key)

    updates, opt_state = opt.update(grads, opt_state, model, value=loss)        ## For reduce on plateau loss accumulation
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, aux_data







#%% Train the model

if train:

    opt = optax.chain(
        optax.clip(config['optimizer']['gradients_lim']),
        optax.adabelief(config['optimizer']['init_lr']),
        optax.contrib.reduce_on_plateau(
            patience= config['optimizer']['on_plateau']['patience'],
            cooldown=config['optimizer']['on_plateau']['cooldown'],
            factor=config['optimizer']['on_plateau']['factor'],
            rtol=config['optimizer']['on_plateau']['rtol'],
            accumulation_size=config['optimizer']['on_plateau']['accum_size'],
            min_scale=config['optimizer']['on_plateau']['min_scale'],
        ),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    ## If a mode file exists, load it and use it
    if os.path.exists(artefacts_folder+"model.eqx"):
        model = eqx.tree_deserialise_leaves(artefacts_folder+"model.eqx", model)
        logger.info("Model found in run folder. Finetuning from these.")
        try:
            with open(artefacts_folder+"opt_state.pkl", 'rb') as f:
                opt_state = pickle.load(f)
        except:
            logger.info("No optimizer state for finetuning. Starting from scratch.")
    else:
        logger.info("No model found in run folder. Training from scratch.")

    losses = []
    med_losses_per_epoch = []
    lr_scales = []

    print_every = config['training']['print_every']
    checkpoint_every = config['training']['checkpoint_every']

    nb_epochs = config['training']['nb_epochs']
    logger.info(f"\n\n=== Beginning training ... ===")
    logger.info(f"  - Number of epochs: {nb_epochs}")
    logger.info(f"  - Number of batches: {trainloader.num_batches}")
    logger.info(f"  - Total number of GD steps: {trainloader.num_batches*nb_epochs}")

    start_time = time.time()

    for epoch in range(nb_epochs):

        losses_epoch = []

        for i, batch in enumerate(trainloader):
            train_key, _ = jax.random.split(train_key)
            model, opt_state, loss, aux = train_step(model, batch, opt_state, train_key)

            losses_epoch.append(loss)
            losses.append(loss)

            lr_scales.append(optax.tree_utils.tree_get(opt_state, "scale"))

        mean_epoch, median_epoch = np.mean(losses_epoch), np.median(losses_epoch)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            logger.info(
                f"Epoch {epoch:-4d}/{nb_epochs:-4d}     Train Loss   -Mean: {mean_epoch:.6f},   -Median: {median_epoch:.6f},   -Latest: {loss:.6f}"
            )

        if epoch%checkpoint_every==0 or epoch==nb_epochs-1:
            eqx.tree_serialise_leaves(checkpoints_folder+f"model_{epoch}.eqx", model)
            np.save(artefacts_folder+"losses.npy", np.array(losses))
            np.save(artefacts_folder+"lr_scales.npy", np.array(lr_scales))

            ## Only save the best model with the lowest mean loss
            med_losses_per_epoch.append(median_epoch)
            if epoch==0 or median_epoch<=np.min(med_losses_per_epoch[:-1]):
                eqx.tree_serialise_leaves(artefacts_folder+"model.eqx", model)
                with open(artefacts_folder+"opt_state.pkl", 'wb') as f:
                    pickle.dump(opt_state, f)
                logger.info("Best model saved ...")

        if epoch==3:     ## Print the output of nvidia-smi to check VRAM usage
            os.system("nvidia-smi")
            os.system("nvidia-smi >> "+artefacts_folder+"training.log")

    wall_time = time.time() - start_time
    logger.info("\nTraining complete. Total time: %d hours %d mins %d secs" %seconds_to_hours(wall_time))

else:
    model = eqx.tree_deserialise_leaves(artefacts_folder+"model.eqx", model)

    try:
        losses = np.load(artefacts_folder+"losses.npy")
        lr_scales = np.load(artefacts_folder+"lr_scales.npy")
    except:
        losses = []

    logger.info(f"Model loaded from {run_folder}model.eqx")








# %% Visualise the training losses

if not os.path.exists(artefacts_folder+"losses.npy"):
    try:
        with open(artefacts_folder+"training.log", 'r') as f:
            lines = f.readlines()
        losses = []
        search_term = "Train Loss (Mean)"
        for line in lines:
            if search_term in line:
                loss = float(line.split(f"{search_term}: ")[1].strip())
                losses.append(loss)
        logger.info("Losses found in the training.log file")
    except:
        logger.info("No losses found in the training.log file")


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 5))

clean_losses = np.array(losses)
epochs = np.arange(len(losses))
loss_name = "NLL" if use_nll_loss else r"$L_2$"
ax = sbplot(epochs, clean_losses, title="Loss History", x_label='Train Steps', y_label=loss_name, ax=ax, y_scale="linear" if use_nll_loss else "log");

clean_losses = np.where(clean_losses<np.percentile(clean_losses, 96), clean_losses, np.nan)
## Plot a second plot with the outliers removed
ax2 = sbplot(epochs, clean_losses, title="Loss History (96th Percentile)", x_label='Train Steps', y_label=loss_name, ax=ax2, y_scale="linear" if use_nll_loss else "log");

plt.draw();
plt.savefig(plots_folder+"loss.png", dpi=100, bbox_inches='tight')

if os.path.exists(artefacts_folder+"lr_scales.npy"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax = sbplot(lr_scales, "g-", title="LR Scales", x_label='Train Steps', ax=ax, y_scale="log");

    # plt.legend()
    plt.draw();
    plt.savefig(plots_folder+"lr_scales.png", dpi=100, bbox_inches='tight')








# %% Other visualisation of the model

if config["model"]["model_type"] == "wsm":

    ## Let's visualise the distribution of values along the main diagonal of A and theta
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].hist(jnp.diag(model.As[0], k=0), bins=100)

    axs[0].set_title("Histogram of diagonal values of A (first layer)")

    axs[1].hist(model.thetas[0], bins=100, label="After Training")
    axs[1].hist(untrained_model.thetas[0], bins=100, alpha=0.5, label="Before Training", color='r')
    axs[1].set_title(r"Histogram of $\theta_0$ values")
    plt.legend();
    plt.draw();
    plt.savefig(plots_folder+"A_theta_histograms.png", dpi=100, bbox_inches='tight')

    ## PLot all values of B in a lineplot (all dimensions)
    if not isinstance(model.Bs[0], eqx.nn.Linear):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(model.Bs[0], label="Values of B")
        ax.set_title("Values of B")
        ax.set_xlabel("Dimension")
        plt.draw();
        plt.savefig(plots_folder+"B_values.png", dpi=100, bbox_inches='tight')

    ## Print the untrained and trained matrices A as imshows with same range
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    min_val = -0.00
    max_val = 0.0003

    img = axs[0].imshow(untrained_model.As[0], cmap='viridis', vmin=min_val, vmax=max_val)
    axs[0].set_title("Untrained A")
    plt.colorbar(img, ax=axs[0], shrink=0.7)

    img = axs[1].imshow(model.As[0], cmap='viridis', vmin=min_val, vmax=max_val)
    axs[1].set_title("Trained A")
    plt.colorbar(img, ax=axs[1], shrink=0.7)
    plt.draw();
    plt.savefig(plots_folder+"A_matrices.png", dpi=100, bbox_inches='tight')


    ## Print the dynamic tanh_params attribute
    if config['model']['apply_dynamic_tanh']:
        latex_string = r"y = \alpha \cdot \text{tanh} \left( \frac{x-b}{a} \right) + \beta"
        logger.info(f"Dynamic tanh params (final root network activation) : ${latex_string}$ ")

        display(Math(latex_string))
        logger.info(f"a, b, alpha, beta: {model.dtanh_params}")

        ## Plot this against a normal tanh
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        x = np.linspace(-3.5, 3.5, 500)
        y = np.tanh(x)
        a, b, alpha, beta = model.dtanh_params
        y2 = alpha * np.tanh((x-b)/a) + beta
        ax.plot(x, y, label="tanh")
        ax.plot(x, y2, label="Dynamic tanh")
        ax.set_title("Dynamic tanh vs tanh after training")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        plt.draw();
        plt.savefig(plots_folder+"dynamic_tanh.png", dpi=100, bbox_inches='tight')


# %% Evaluate the model on the test set

@eqx.filter_jit
def eval_step(model, X, times, key, inference_start=None):
    """ Evaluate the model on a batch of data. """
    X_recons, _ = model(X, times, key, inference_start)
    return X_recons

def eval_on_test_set(model, key):
    mses = []
    new_key, _ = jax.random.split(key)
    for i, batch in enumerate(testloader):
        new_key, _ = jax.random.split(new_key)
        (X_true, times), X_labs_outs = batch

        X_recons = eval_step(model, X_true, times, new_key, inference_start=None)
        if use_nll_loss:
            X_recons = X_recons[:, :, :data_size]
        if dataset in repeat_datasets:
            mse = jnp.mean((X_recons - X_labs_outs)**2)
        else:
            mse = jnp.mean((X_recons - X_true)**2)
        mses.append(mse)

    return np.mean(mses), np.median(mses), np.min(mses)

test_key, _ = jax.random.split(train_key)
mean_mse, median_mse, min_mse = eval_on_test_set(model, test_key)

logger.info("Evaluation of MSE on the test set, at the end of the training (Current Best Model):")
logger.info(f"    - Mean : {mean_mse:.6f}")
logger.info(f"    - Median : {median_mse:.6f}")
logger.info(f"    - Min : {min_mse:.6f}")

nb_epochs = config['training']['nb_epochs']
checkpoint_every = config['training']['checkpoint_every']

best_model = model
best_mse = mean_mse
best_mse_epoch = nb_epochs-1        ## TODO: model might not have been trained for all epochs

if os.path.exists(artefacts_folder+"test_mses.npz"):
    checkpoints_data = np.load(artefacts_folder+"test_mses.npz")
    mses_chekpoints = checkpoints_data['data']
    best_mse_epoch = checkpoints_data['best_epoch'].item()
    best_mse = checkpoints_data['best_mse'].item()
    best_model = eqx.tree_deserialise_leaves(checkpoints_folder+f"model_{best_mse_epoch}.eqx", model)
    id_checkpoints = (np.arange(0, nb_epochs, checkpoint_every).tolist() + [nb_epochs-1])[:len(mses_chekpoints)]
    logger.info("Checkpoints MSE artefact file found. Loading it.")

else:
    mses_chekpoints = [] 
    id_checkpoints = []

    ## Lead the model at each checkpoint and evaluate it
    for i in list(range(0, nb_epochs, checkpoint_every))+[nb_epochs-1]:
        try:
            model = eqx.tree_deserialise_leaves(checkpoints_folder+f"model_{i}.eqx", model)
        except:
            logger.info(f"Checkpoint {i} not found. Skipping.")
            continue

        mean, med, min_ = eval_on_test_set(model, test_key)
        mses_chekpoints.append(mean)
        id_checkpoints.append(i)

        if mean<best_mse:
            best_model = model
            best_mse = mean
            best_mse_epoch = i
            logger.info(f"New best model found at epoch {i} with MSE: {best_mse:.6f}")
        # logger.info(f"Checkpoint {i} MSE: {mean:.6f} (Mean), {med:.6f} (Median), {min_:.6f} (Min)")

    ## Save the checkpoints MSEs artefacts
    np.savez(artefacts_folder+"test_mses.npz", data=np.array(mses_chekpoints), best_epoch=best_mse_epoch, best_mse=best_mse)

## Plot the MSE of the checkpoints
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax = sbplot(id_checkpoints, mses_chekpoints, title="MSE on Test Set at Various Checkpoints", x_label='Epoch', y_label='MSE', ax=ax, y_scale="log", linewidth=3);
plt.axvline(x=best_mse_epoch, color='r', linestyle='--', linewidth=3, label=f"Best MSE: {best_mse:.6f} at Epoch {best_mse_epoch}")
plt.legend(fontsize=16)
plt.draw();
plt.savefig(plots_folder+"checkpoints_mse.png", dpi=100, bbox_inches='tight')
logger.info(f"Best model found at epoch {best_mse_epoch} with MSE: {best_mse:.6f}")


### ===== Very importtant: Set the best model on test set as the model for visualisation ? ==== TODO
# model = best_model

# %% Visualising a few reconstruction samples

## Set inference mode to True
visloader = NumpyLoader(testloader.dataset, batch_size=16, shuffle=True)

nb_cols = 3 if use_nll_loss else 2
fig, axs = plt.subplots(4, 4*nb_cols, figsize=(16*3, 16), sharex=True, constrained_layout=True)

batch = next(iter(visloader))
(xs_true, times), labels = batch

inference_start = config['training']['inference_start']
xs_recons = eval_step(model=model, 
                      X=xs_true, 
                      times=times, 
                      key=test_key, 
                      inference_start=inference_start)

if use_nll_loss:
    xs_uncert = xs_recons[:, :, data_size:]
    xs_recons = xs_recons[:, :, :data_size]

res = (width, width, data_size)
mpl.rcParams['lines.linewidth'] = 3

for i in range(4):
    for j in range(4):
        x = xs_true[i*4+j]
        x_recons = xs_recons[i*4+j]
        x_full = labels[i*4+j]

        if dataset in dynamics_datasets+repeat_datasets:
            ## Min/max along dim0, for both x and x_recons
            min_0, max_0 = min(np.min(x[:, dim0]), np.min(x_recons[:, dim0])), max(np.max(x[:, dim0]), np.max(x_recons[:, dim0]))
            min_1, max_1 = min(np.min(x[:, dim1]), np.min(x_recons[:, dim1])), max(np.max(x[:, dim1]), np.max(x_recons[:, dim1]))
            eps = 0.04

        if dataset in image_datasets:
            to_plot = x.reshape(res)
            if dataset=="celeba":
                to_plot = (to_plot + 1) / 2
            axs[i, nb_cols*j].imshow(to_plot, cmap='gray')
        elif dataset == "trends":
            axs[i, nb_cols*j].plot(x, color=colors[labels[i*4+j]])
        elif dataset in repeat_datasets:
            axs[i, nb_cols*j].set_ylim([min_1-eps, max_1+eps])
            axs[i, nb_cols*j].plot(x_full[:, dim0], color=colors[(i*4+j)%len(colors)])
            axs[i, nb_cols*j].plot(x_full[:, dim1], color=colors[(i*4+j)%len(colors)], linestyle='-.')
        else:
            # axs[i, nb_cols*j].set_xlim([min_0-eps, max_0+eps])
            axs[i, nb_cols*j].set_ylim([min_1-eps, max_1+eps])
            axs[i, nb_cols*j].plot(x[:, dim0], color=colors[labels[i*4+j]%len(colors)])
            axs[i, nb_cols*j].plot(x[:, dim1], color=colors[labels[i*4+j]%len(colors)], linestyle='-.')
        if i==0:
            axs[i, nb_cols*j].set_title("GT", fontsize=40)
        # axs[i, nb_cols*j].axis('off')

        if dataset in image_datasets:
            to_plot = x_recons.reshape(res)
            if dataset=="celeba":
                to_plot = (to_plot + 1) / 2
            axs[i, nb_cols*j+1].imshow(to_plot, cmap='gray')
        elif dataset in dynamics_datasets and dataset not in repeat_datasets:
            # axs[i, nb_cols*j+1].set_xlim([min_0-eps, max_0+eps])
            axs[i, nb_cols*j+1].set_ylim([min_1-eps, max_1+eps])
            axs[i, nb_cols*j+1].plot(x_recons[:, dim0], color=colors[labels[i*4+j]%len(colors)])
            axs[i, nb_cols*j+1].plot(x_recons[:, dim1], color=colors[labels[i*4+j]%len(colors)], linestyle='-.')
        elif dataset in repeat_datasets:
            axs[i, nb_cols*j+1].set_ylim([min_1-eps, max_1+eps])
            axs[i, nb_cols*j+1].plot(x_recons[:, dim0], color=colors[(i*4+j)%len(colors)])
            axs[i, nb_cols*j+1].plot(x_recons[:, dim1], color=colors[(i*4+j)%len(colors)], linestyle='-.')
        else:
            axs[i, nb_cols*j+1].plot(x_recons, color=colors[labels[i*4+j]])
        if i==0:
            axs[i, nb_cols*j+1].set_title("Recons", fontsize=40)
        # axs[i, nb_cols*j+1].axis('off')

        if use_nll_loss:
            logger.info(f"Min/Max Uncertainty: {np.min(xs_uncert):.3f}, {np.max(xs_uncert):.3f}")
            if dataset in image_datasets:
                to_plot = xs_uncert[i*4+j].reshape(res)
                axs[i, nb_cols*j+2].imshow(to_plot, cmap='gray')
            elif dataset in dynamics_datasets:
                to_plot = xs_uncert[i*4+j]
                axs[i, nb_cols*j+2].plot(to_plot[:, dim0], to_plot[:, dim1], color=colors[labels[i*4+j]%len(colors)])

            if i==0:
                axs[i, nb_cols*j+2].set_title("Uncertainty", fontsize=36)
            # axs[i, nb_cols*j+2].axis('off')

plt.suptitle(f"Reconstruction using {inference_start} initial steps", fontsize=65)
plt.draw();
plt.savefig(plots_folder+"samples_generated.png", dpi=100, bbox_inches='tight')





# %% Analysing the weight space
### 

@eqx.filter_jit
def eval_step(model, X, times, key, inference_start=None):
    """ Evaluate the model on a batch of data. """
    X_recons, thetas = model(X, times, key, inference_start)
    return X_recons, thetas

thetas = eval_step(model=model,
                    X=xs_true, 
                    times=times, 
                    key=test_key, 
                    inference_start=inference_start)[1]
print(f"Shape of thetas: {thetas.shape}")

def apply_theta(theta):
    shapes, treedef, static, _ = model.root_utils[0]
    params = unflatten_pytree(theta, shapes, treedef)
    root_fun = eqx.combine(params, static)
    ts = jnp.linspace(0, 1, seq_length)[:, None]
    X_recons = eqx.filter_vmap(root_fun)(ts)
    return X_recons

results_matrix = eqx.filter_vmap(apply_theta)(thetas[0])


#%%
results_matrix.shape        ## (256, 256, 2)

## PLot the results matrix in an imshow, on two axes
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
min_val = None
max_val = None
cmap = "grey"
fontcolor = 'r'
img = axs[0].imshow(results_matrix[:, :, 0], cmap=cmap, origin='upper')
axs[0].set_title("Displacement", fontsize=40)
plt.colorbar(img, ax=axs[0], shrink=0.7)
img = axs[1].imshow(results_matrix[:, :, 1], cmap=cmap, origin='upper')            
axs[1].set_title("Velocity", fontsize=40)
plt.colorbar(img, ax=axs[1], shrink=0.7)

## Set x ticks as linspace from 0 to 1
ticks = np.linspace(0, 1, 5)
axs[0].set_xticks(ticks * 255)
axs[1].set_xticks(ticks * 255)
axs[0].set_xticklabels([f"{tick:.2f}" for tick in ticks], fontsize=26)
axs[1].set_xticklabels([f"{tick:.2f}" for tick in ticks], fontsize=26)

axs[0].set_xlabel(r"Time $\tau$", fontsize=40)
axs[1].set_xlabel(r"Time $\tau$", fontsize=40)


## Set the y ticks as the powers of 2
yticks = np.array([1, 64, 128, 192, 256])
axs[0].set_yticks(yticks-1)
axs[0].set_yticklabels([f"{tick:.0f}" for tick in yticks-1], fontsize=26)


# axs[0].set_xlabel(r"Time Step $t$")
# axs[1].set_xlabel(r"Time Step $t$")
axs[0].set_ylabel(r"$\theta_t$", fontsize=40)
# axs[1].set_ylabel(r"$\theta_t$")

## Draw a oblique line for the decoding direction
axs[0].arrow(16, 16, 220, 220, head_width=10, head_length=10, fc=fontcolor, ec=fontcolor, lw=6)
axs[0].text(128+15, 128-10, r"$\theta_t(\tau)$", fontsize=40, color=fontcolor, rotation=-45, ha='center', va='center')

axs[1].arrow(16, 16, 220, 220, head_width=10, head_length=10, fc=fontcolor, ec=fontcolor, lw=6)
axs[1].text(128+15, 128-10, r"$\theta_t(\tau)$", fontsize=40, color=fontcolor, rotation=-45, ha='center', va='center')

plt.draw();
plt.savefig(plots_folder+"results_matrix.png", dpi=100, bbox_inches='tight')
plt.savefig(plots_folder+"results_matrix.pdf", dpi=100, bbox_inches='tight')


#%% Visualise the thetas themselves
fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharey=True)


## Noramlisation along the axis1
min_val, max_val = np.min(thetas[0, :, :], axis=0, keepdims=True), np.max(thetas[0, :, :], axis=0, keepdims=True)
normed_theta_0 = (thetas[0, :, :] - min_val) / (max_val - min_val)

print(f"Min/Max of thetas: {np.min(thetas[0, :, :])}, {np.max(thetas[0, :, :])}")
print(f"Min/Max of normed thetas: {np.min(normed_theta_0)}, {np.max(normed_theta_0)}")

print(f"Shape of thetas: {thetas[0, :, :].shape}, {normed_theta_0.shape}")
img = ax.imshow(normed_theta_0.T[::10], cmap='coolwarm', origin='upper', vmin=None, vmax=None, interpolation=None)

## Plot color bar
plt.colorbar(img, ax=ax, shrink=1.0)
ax.set_title(r"$\theta_t$", fontsize=40)




#%% Let's finetune theta0, but each step individually




#%% Let's finetune theta0 for 256 gradient descent steps on the trajectory
def tinetune_loss_fn(thet):
    X_recons = apply_theta(thet)
    print(f"Shape of X_recons: {X_recons.shape}, {xs_true[0].shape}")
    loss_r = optax.l2_loss(X_recons[...,:1], xs_true[0])
    loss = jnp.mean(loss_r)
    return loss

finetune_opt = optax.adabelief(1e-2)
# thet = thetas[0, 0, :]
finetune_key = jax.random.PRNGKey(time.time_ns()%100000)
# thet = thetas[0, 0, :] + jax.random.uniform(model_key, thetas[0, -1, :].shape, minval=-0.1, maxval=0.1)
# thet = jax.random.uniform(model_key, thetas[0, -1, :].shape, minval=-1., maxval=1.)

## Initialis a normal root network
new_root = model.root_utils[0][-1]
input_dim, output_dim, width_size, depth, activation = new_root
print(f"Shape of new_root: {new_root}")


nb_runs = 1

finetune_thetas_all_all = []
for i in range(nb_runs):
    finetune_key, _ = jax.random.split(finetune_key)

    root_thet = eqx.nn.MLP(input_dim, output_dim, width_size, depth, activation=activation, key=finetune_key)
    params, static = eqx.partition(root_thet, eqx.is_array)
    thet, _, _ = flatten_pytree(params)

    ## Same initialisation, be carefull
    thet = thetas[0, 0, :]

    finetune_opt_state = finetune_opt.init(eqx.filter(thet, eqx.is_array))

    @eqx.filter_jit
    def finetune_step(thet, opt_state):
        print('\nCompiling function "finetune_step" ...')
        loss, grads = eqx.filter_value_and_grad(tinetune_loss_fn)(thet)
        updates, opt_state = finetune_opt.update(grads, opt_state, thet)
        thet = eqx.apply_updates(thet, updates)
        return thet, opt_state, loss

    finetune_losses = []
    finetune_thetas = []
    for ep in range(256):
        thet, finetune_opt_state, loss = finetune_step(thet, finetune_opt_state)
        finetune_losses.append(loss)
        finetune_thetas.append(thet)

        if ep%100==0 or ep==255:
            logger.info(f"Finetuning step {ep} - Loss: {loss:.6f}")


    finetune_thetas = np.stack(finetune_thetas)

    finetune_thetas_all_all.append(finetune_thetas)

# finetune_thetas_all_all = np.stack(finetune_thetas_all_all)

## Plot the finetuning losses
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(finetune_losses, 'b-', lw=2, label="Finetuning Loss")
ax.set_title("Finetuning Loss")
ax.set_xlabel("Finetuning Steps")

## Visualize the finetuning trajectory
fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharey=True)
# finetune_recons = apply_theta(thetas[0, 0, :])
finetune_recons = apply_theta(thet)
ax.plot(finetune_recons[:, 0])
ax.plot(finetune_recons[:, 1])
ax.set_title("Finetuning Trajectory")


#%% Let's do PCA on theta_0


from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
import umap

# pca = PCA(n_components=2)
# pca = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
pca = umap.UMAP(n_components=2, random_state=145)

# pca.fit(thetas[0, :, :])
# pca.fit(finetune_thetas)
# pca.fit(np.concatenate([thetas[0, :, :], finetune_thetas], axis=0))
pca.fit(np.concatenate([thetas[0, :, :]]+finetune_thetas_all_all, axis=0))

# print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
# print(f"Components: {pca.components_}")

## transform and plot the data
X_pca = pca.transform(thetas[0, :, :])

## Transform the finetuned thetas as well
finetune_thetas_pca = pca.transform(finetune_thetas)


# fig, (ax2, ax) = plt.subplots(2, 1, figsize=(10, 10))
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
# ax.plot(X_pca[:, 0], X_pca[:, 1], 'bo', lw=2)

## Earlier Xs {id=0} should be different from latter ones {id=256}. Increased the alpha
for i in range(256):
# for i in range(100, 256):
    ## calculate alpha and rescale it into 0.25, 1
    alpha = (i+1)/256
    markersize = ((i+1)/256)*20

    # index = (i+1)/256
    # alpha = 0.25 + (1-0.25)*index
    # markersize = 5 + (20-5)*index

    ax.plot(X_pca[i, 0], X_pca[i, 1], 'bo', lw=2, alpha=alpha, markersize=markersize, label="WARP" if i==255 else None)

    ax.plot(finetune_thetas_pca[i, 0], finetune_thetas_pca[i, 1], 'ro', lw=2, alpha=alpha, markersize=markersize, label="Gradient Descent" if i==255 else None)


# # print(finetune_thetas_all_all.shape)
# ## Do the PCA for the other runs as well
# for i in range(0, nb_runs-1):
#     finetune_thetas_pca = pca.transform(finetune_thetas_all_all[i])
#     ax.plot(finetune_thetas_pca[:, 0], finetune_thetas_pca[:, 1], 'ro', lw=2, alpha=alpha, markersize=markersize)

# ## PLot the initial and final points
# ax.plot(X_pca[0, 0], X_pca[0, 1], 'rX', lw=2, alpha=1, label="Initial", markersize=20)
# ax.plot(X_pca[255, 0], X_pca[255, 1], 'gX', lw=2, alpha=1, label="Final", markersize=40)

# ax.legend(fontsize=30, loc='upper left')

## make th legend horionztal, higher than the top of the plot
ax.legend(fontsize=30, loc='upper left', bbox_to_anchor=(-0.075, 1.20), ncol=2)

# ax.set_title(r"PCA of $\theta_t$", fontsize=40)
ax.set_xlabel(r"$PC_1$", fontsize=40)
ax.set_ylabel(r"$PC_2$", fontsize=40)

plt.draw();
plt.savefig(plots_folder+"wsm_vs_gd_traj.png", dpi=100, bbox_inches='tight')
plt.savefig(plots_folder+"wsm_vs_gd_traj.pdf", dpi=100, bbox_inches='tight')


fig, ax2 = plt.subplots(1, 1, figsize=(10, 5))

## %% Visulualize the the difference betwen the weights, i.e. the gradient norms
## Calculate the difference between the weights
diff_warp = np.mean(np.abs(thetas[0, 1:, :]-thetas[0, :-1, :]), axis=1)
diff_gd = np.mean(np.abs(finetune_thetas[1:, :]-finetune_thetas[:-1, :]), axis=1)

min_val = np.min(np.concatenate([diff_warp, diff_gd]))
max_val = np.max(np.concatenate([diff_warp, diff_gd]))

ax2.plot(diff_warp, 'bo', lw=2, alpha=0.5, markersize=5, label="WARP")
ax2.plot(diff_gd, 'ro', lw=2, alpha=0.5, markersize=5, label="Gradient Descent")
ax2.set_ylim([min_val, max_val])
ax2.set_yscale("log")

ax2.set_xlabel(r"Time Step $t$", fontsize=30)
ax2.set_ylabel(r"$|\theta_t - \theta_{t-1}|$", fontsize=20)
ax2.legend(fontsize=20, loc='upper right')

plt.draw();
plt.savefig(plots_folder+"wsm_vs_gd_grad.png", dpi=100, bbox_inches='tight')
plt.savefig(plots_folder+"wsm_vs_gd_grad.pdf", dpi=100, bbox_inches='tight')

# %%

#%%[markdown]

# ## Generative Recurrent Neural Networks in weight space

#%%

# %load_ext autoreload
# %autoreload 2

from utils import *
from loaders import *
from models import *

import jax
print("Available devices:", jax.devices())

from jax import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)
# from jax.experimental import checkify

## Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import jax.numpy as jnp
import equinox as eqx
import optax
import time

import seaborn as sb
sb.set_context("poster")

import yaml
import argparse
import numpy as np
import torch
import jax
import os
import time
from pprint import pprint
import sys









#%%

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

if _in_ipython_session:
    args = argparse.Namespace(config_file='config.yaml')
    print("Notebook Session: Using default config.yaml file")

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
        ## Complain about the usage and exit
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
data_folder = config['general']['data_folder']

### Create and setup the run and data folders
if train:
    run_folder = make_run_folder('./runs/')
else:
    run_folder = "./"
    data_folder = f"../../{data_folder}/"

print("Using run folder:", run_folder)
checkpoints_folder, logger = setup_run_folder(run_folder, training=train)

## Print the config file using the logger and pprint
logger.info(f"Config file: {args.config_file}")
logger.info("Config file content:")
pprint(config)









#%%

trainloader, testloader, data_props = make_dataloaders(config)
nb_classes, seq_length, data_size, width = data_props

batch = next(iter(testloader))
(images, times), labels = batch
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Seq length:", seq_length)
print("Data size:", data_size)

print("Min and Max in the dataset:", jnp.min(images), jnp.max(images))
print("==Number of batches:")
print("  - Train:", trainloader.num_batches)
print("  - Test:", testloader.num_batches)

## Plot a few samples, along with their labels as title in a 4x4 grid (chose them at random)
fig, axs = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
colors = ['r', 'g', 'b', 'c', 'm', 'y']

dataset = config['general']['dataset']
image_datasets = ["mnist", "mnist_fashion", "cifar", "celeba"]
dynamics_datasets = ["lorentz63", "lorentz96", "lotka", "trends"]

res = (width, width, data_size)
for i in range(4):
    for j in range(4):
        idx = np.random.randint(0, images.shape[0])
        if dataset in image_datasets:
            to_plot = images[idx].reshape(res)
            if dataset=="celeba":
                to_plot = (to_plot + 1) / 2
            axs[i, j].imshow(to_plot, cmap='gray')
        elif dataset=="trends":
            axs[i, j].plot(images[idx], color=colors[labels[idx]])
            dim0, dim1 = (0, 1)
        else:
            axs[i, j].plot(images[idx, :, dim0], images[idx, :, dim1], color=colors[labels[idx]%len(colors)])

        axs[i, j].set_title(f"Class: {labels[idx]}", fontsize=12)
        axs[i, j].axis('off')


# %% Define the model and loss function

model_key, train_key = jax.random.split(main_key, num=2)
model = make_model(model_key, config)
untrained_model = model

nb_recons_loss_steps = config['training']['nb_recons_loss_steps']
use_nll_loss = config['training']['use_nll_loss']

def loss_fn(model, batch, key):
    """ Loss function for the model. A batch contains: (Xs, Ts), Ys
    Xs: (batch, time, data_size)
    Ts: (batch, time)
    Ys: (batch, num_classes)
    """
    (X_true, times), _ = batch

    X_recons = model(X_true, times, key, inference_start=None)

    if nb_recons_loss_steps !=-1:  ## Use all the steps
        ## Randomly sample some steps in the sequence for the loss
        batch_size, nb_timesteps = X_true.shape[0], X_true.shape[1]
        indices_0 = jnp.arange(batch_size)
        indices_1 = jax.random.randint(key, (batch_size, nb_recons_loss_steps), 0, nb_timesteps)

        X_recons_ = jnp.stack([X_recons[indices_0, indices_1[:,j]] for j in range(nb_recons_loss_steps)], axis=1)
        X_true_ = jnp.stack([X_true[indices_0, indices_1[:,j]] for j in range(nb_recons_loss_steps)], axis=1)

    else:
        X_recons_ = X_recons
        X_true_ = X_true

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
        optax.clip(config['optimizer']['gradient_lim']),
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

    losses = []
    mean_losses_per_epoch = []
    lr_scales = []

    print_every = config['training']['print_every']
    checkpoint_every = config['training']['checkpoint_every']

    nb_epochs = config['training']['nb_epochs']
    logger.info(f"\n\n=== Beginning Training ... ===")
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
        mean_losses_per_epoch.append(mean_epoch)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            logger.info(
                f"Epoch {epoch:-5d}/{nb_epochs:-5d}, "
                f"Train Loss (Mean): {mean_epoch:.6f}, "
                f"Train Loss (Median): {median_epoch:.6f}, "
                f"Train Loss (Latest): {loss:.6f}, "
            )

        if epoch%checkpoint_every==0 or epoch==nb_epochs-1:
            eqx.tree_serialise_leaves(checkpoints_folder+f"model_{epoch}.eqx", model)
            np.save(run_folder+"losses.npy", np.array(losses))
            np.save(run_folder+"lr_scales.npy", np.array(lr_scales))

            ## Only save the best model with the lowest mean loss
            if epoch>0 and mean_epoch<min(mean_losses_per_epoch[:-1]):
                eqx.tree_serialise_leaves(run_folder+"model.eqx", model)

    wall_time = time.time() - start_time
    print("\nTraining complete. Total time: %d hours %d mins %d secs" %seconds_to_hours(wall_time))

else:
    model = eqx.tree_deserialise_leaves(run_folder+"model.eqx", model)

    try:
        losses = np.load(run_folder+"losses.npy")
        lr_scales = np.load(run_folder+"lr_scales.npy")
    except:
        losses = []

    print(f"Model loaded from {run_folder}model.eqx")








# %% Visualise the training losses

if not os.path.exists(run_folder+"losses.npy"):
    try:
        with open(run_folder+"training.log", 'r') as f:
            lines = f.readlines()
        losses = []
        search_term = "Train Loss (Mean)"
        for line in lines:
            if search_term in line:
                loss = float(line.split(f"{search_term}: ")[1].strip())
                losses.append(loss)
        print("Losses found in the training.log file")
    except:
        print("No losses found in the nohup.log file")


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 4))

clean_losses = np.array(losses)
epochs = np.arange(len(losses))

ax = sbplot(epochs, clean_losses, label="Loss History", x_label='Train Steps', y_label='Loss', ax=ax, dark_background=False, y_scale="linear" if use_nll_loss else "log");

clean_losses = np.where(clean_losses<np.percentile(clean_losses, 96), clean_losses, np.nan)
## Plot a second plot with the outliers removed
ax2 = sbplot(epochs, clean_losses, label="Loss History (96th Percentile)", x_label='Train Steps', y_label='Loss', ax=ax2, dark_background=False, y_scale="linear" if use_nll_loss else "log");

plt.legend()
plt.draw();
plt.savefig(run_folder+"loss.png", dpi=100, bbox_inches='tight')


if os.path.exists(run_folder+"lr_scales.npy"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax = sbplot(lr_scales, "g-", label="Learning Rate Scales", x_label='Train Steps', y_label='LR Scale', ax=ax, dark_background=False, y_scale="log");

    plt.legend()
    plt.draw();
    plt.savefig(run_folder+"lr_scales.png", dpi=100, bbox_inches='tight')








# %% Other visualisation of the model

## Let's visualise the distribution of values along the main diagonal of A and theta
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].hist(jnp.diag(model.As[0], k=0), bins=100)

axs[0].set_title("Histogram of diagonal values of A (first layer)")

axs[1].hist(model.thetas[0], bins=100, label="After Training")
axs[1].hist(untrained_model.thetas[0], bins=100, alpha=0.5, label="Before Training", color='r')
axs[1].set_title(r"Histogram of $\theta_0$ values")
plt.legend();
plt.draw();
plt.savefig(run_folder+"A_theta_histograms.png", dpi=100, bbox_inches='tight')

## PLot all values of B in a lineplot (all dimensions)
if not isinstance(model.Bs[0], eqx.nn.Linear):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(model.Bs[0], label="Values of B")
    ax.set_title("Values of B")
    ax.set_xlabel("Dimension")
    plt.draw();
    plt.savefig(run_folder+"B_values.png", dpi=100, bbox_inches='tight')

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
plt.savefig(run_folder+"A_matrices.png", dpi=100, bbox_inches='tight')








# %% Evaluate the model on the test set

@eqx.filter_jit
def eval_step(model, X, times, key, inference_start=None):
    """ Evaluate the model on a batch of data. """
    X_recons = model(X, times, key, inference_start)
    return X_recons

mses = []
test_key, _ = jax.random.split(train_key)
for i, batch in enumerate(testloader):
    test_key, _ = jax.random.split(test_key)
    (X_true, times), X_labels = batch

    X_recons = eval_step(model, X_true, times, test_key, inference_start=None)
    if use_nll_loss:
        X_recons = X_recons[:, :, :data_size]
    mse = jnp.mean((X_recons - X_true)**2)
    mses.append(mse)

print("Evaluation of MSE the test set:")
print(f"    - Mean : {np.mean(mses):.6f}")
print(f"    - Median : {np.median(mses):.6f}")
print(f"    - Min : {np.min(mses):.6f}")






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
for i in range(4):
    for j in range(4):
        x = xs_true[i*4+j]
        x_recons = xs_recons[i*4+j]

        if dataset in image_datasets:
            to_plot = x.reshape(res)
            if dataset=="celeba":
                to_plot = (to_plot + 1) / 2
            axs[i, nb_cols*j].imshow(to_plot, cmap='gray')
        elif dataset == "trends":
            axs[i, nb_cols*j].plot(x, color=colors[labels[i*4+j]])
        else:
            axs[i, nb_cols*j].plot(x[:, dim0], x[:, dim1], color=colors[labels[i*4+j]%len(colors)])
        if i==0:
            axs[i, nb_cols*j].set_title("GT", fontsize=40)
        axs[i, nb_cols*j].axis('off')

        if dataset in image_datasets:
            to_plot = x_recons.reshape(res)
            if dataset=="celeba":
                to_plot = (to_plot + 1) / 2
            axs[i, nb_cols*j+1].imshow(to_plot, cmap='gray')
        elif dataset in dynamics_datasets:
            axs[i, nb_cols*j+1].plot(x_recons[:, dim0], x_recons[:, dim1], color=colors[labels[i*4+j]%len(colors)])
        else:
            axs[i, nb_cols*j+1].plot(x_recons, color=colors[labels[i*4+j]])
        if i==0:
            axs[i, nb_cols*j+1].set_title("Recons", fontsize=40)
        axs[i, nb_cols*j+1].axis('off')

        if use_nll_loss:
            if dataset in image_datasets:
                to_plot = xs_uncert[i*4+j].reshape(res)
                print("Min and Max of the uncertainty:", jnp.min(to_plot), jnp.max(to_plot))
                if dataset=="celeba":
                    to_plot = (to_plot + 1) / 2
                    axs[i, nb_cols*j+2].imshow(to_plot, cmap='gray')
            elif dataset in dynamics_datasets:
                to_plot = xs_uncert[i*4+j]
                print("Min and max of the uncertainty:", jnp.min(to_plot), jnp.max(to_plot))
                axs[i, nb_cols*j+2].plot(to_plot[:, dim0], to_plot[:, dim1], color=colors[labels[i*4+j]%len(colors)])

            if i==0:
                axs[i, nb_cols*j+2].set_title("Uncertainty", fontsize=36)
            axs[i, nb_cols*j+2].axis('off')

plt.suptitle(f"Reconstruction using {inference_start} initial pixels", fontsize=65)
plt.draw();
plt.savefig(run_folder+"reconstruction.png", dpi=100, bbox_inches='tight')


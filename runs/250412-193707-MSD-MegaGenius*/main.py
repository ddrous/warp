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

import yaml
import argparse
import os
import time
from pprint import pprint
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
logger, checkpoints_folder, plots_folder, artefacts_folder = setup_run_folder(run_folder, training=train)

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
(images, times), labels = batch
logger.info(f"Images shape: {images.shape}")
logger.info(f"Labels shape: {labels.shape}")
logger.info(f"Seq length: {seq_length}")
logger.info(f"Data size: {data_size}")
logger.info(f"Min/Max in the dataset: {np.min(images), np.max(images)}")
logger.info("Number of batches:")
logger.info(f"  - Train: {trainloader.num_batches}")
logger.info(f"  - Test: {testloader.num_batches}")

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
            dim0, dim1 = (0, 1)
            axs[i, j].plot(images[idx, :, dim0], images[idx, :, dim1], color=colors[labels[idx]%len(colors)])

        axs[i, j].set_title(f"Class: {labels[idx]}", fontsize=12)
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
    (X_true, times), _ = batch

    X_recons = model(X_true, times, key, inference_start=None)

    if nb_recons_loss_steps is not None:  ## Use all the steps
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
max_val = 0.00003

img = axs[0].imshow(untrained_model.As[0], cmap='viridis', vmin=min_val, vmax=max_val)
axs[0].set_title("Untrained A")
plt.colorbar(img, ax=axs[0], shrink=0.7)

img = axs[1].imshow(model.As[0], cmap='viridis', vmin=min_val, vmax=max_val)
axs[1].set_title("Trained A")
plt.colorbar(img, ax=axs[1], shrink=0.7)
plt.draw();
plt.savefig(plots_folder+"A_matrices.png", dpi=100, bbox_inches='tight')








# %% Evaluate the model on the test set

@eqx.filter_jit
def eval_step(model, X, times, key, inference_start=None):
    """ Evaluate the model on a batch of data. """
    X_recons, thetas = model(X, times, key, inference_start)
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

logger.info("Evaluation of MSE the test set:")
logger.info(f"    - Mean : {np.mean(mses):.6f}")
logger.info(f"    - Median : {np.median(mses):.6f}")
logger.info(f"    - Min : {np.min(mses):.6f}")






# %% Visualising a few reconstruction samples

## Change the numpy and torch seed
np.random.seed(seed+1)
torch.manual_seed(seed+1)

## Set inference mode to True
visloader = NumpyLoader(testloader.dataset, batch_size=16, shuffle=True)

nb_cols = 3 if use_nll_loss else 2
fig, axs = plt.subplots(4, 4*nb_cols, figsize=(16*3, 16), sharex=False, constrained_layout=True)

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
        else:
            # axs[i, nb_cols*j].set_xlim([min_0-eps, max_0+eps])
            axs[i, nb_cols*j].set_ylim([min_1-eps, max_1+eps])
            axs[i, nb_cols*j].plot(x[:, dim0], color=colors[labels[i*4+j]%len(colors)])
            axs[i, nb_cols*j].plot(x[:, dim1], color=colors[labels[i*4+j]%len(colors)])
        if i==0:
            axs[i, nb_cols*j].set_title("GT", fontsize=40)
        # axs[i, nb_cols*j].axis('off')

        if dataset in image_datasets:
            to_plot = x_recons.reshape(res)
            if dataset=="celeba":
                to_plot = (to_plot + 1) / 2
            axs[i, nb_cols*j+1].imshow(to_plot, cmap='gray')
        elif dataset in dynamics_datasets:
            # axs[i, nb_cols*j+1].set_xlim([min_0-eps, max_0+eps])
            axs[i, nb_cols*j+1].set_ylim([min_1-eps, max_1+eps])
            axs[i, nb_cols*j+1].plot(x_recons[:, dim0], color=colors[labels[i*4+j]%len(colors)])
            axs[i, nb_cols*j+1].plot(x_recons[:, dim1], color=colors[labels[i*4+j]%len(colors)])
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




#%%
len(model.root_utils[0])


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
# min_val = 0.
# max_val = 1.


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

# def tinetune_loss_fn(thets):
#     X_recons = eqx.filter_vmap(apply_theta)(thets)
#     X_recons = jnp.stack([X_recons[i, i, :] for i in range(X_recons.shape[0])], axis=0)
#     print(f"Shape of X_recons: {X_recons.shape}")
#     loss_r = optax.l2_loss(X_recons, xs_true[0])
#     loss = jnp.mean(loss_r)
#     return loss

# finetune_opt = optax.adabelief(1e-2)
# # thet = thetas[0, :, :]
# thet = jnp.tile(thetas[0, 0, :], (256, 1))
# print(f"Shape of thet: {thet.shape}")

# finetune_opt_state = finetune_opt.init(eqx.filter(thet, eqx.is_array))

# @eqx.filter_jit
# def finetune_step(thet, opt_state):
#     print('\nCompiling function "finetune_step" ...')
#     loss, grads = eqx.filter_value_and_grad(tinetune_loss_fn)(thet)
#     updates, opt_state = finetune_opt.update(grads, opt_state, thet)
#     thet = eqx.apply_updates(thet, updates)
#     return thet, opt_state, loss

# finetune_losses = []
# finetune_thetas_all = []
# for ep in range(256):
#     thet, finetune_opt_state, loss = finetune_step(thet, finetune_opt_state)
#     finetune_losses.append(loss)
#     finetune_thetas_all.append(thet)

#     if ep%100==0 or ep==255:
#         logger.info(f"Finetuning step {ep} - Loss: {loss:.6f}")


# finetune_thetas_all = np.stack(finetune_thetas_all)
# ## Plot the finetuning losses
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# ax.plot(finetune_losses, 'b-', lw=2, label="Finetuning Loss")
# ax.set_title("Finetuning Loss")
# ax.set_xlabel("Finetuning Steps")


# ## Visualize the finetuning trajectory
# fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharey=True)
# # min_val = 0.
# # max_val = 1.
# # finetune_recons = apply_theta(thetas[0, 0, :])
# X_recons = eqx.filter_vmap(apply_theta)(thet)
# finetune_recons = jnp.stack([X_recons[i, i, :] for i in range(X_recons.shape[0])], axis=0)
# ax.plot(finetune_recons[:, 0])
# ax.plot(finetune_recons[:, 1])
# ax.set_title("Finetuning Trajectory")
# #%%
# # finetune_thetas = finetune_thetas_all[-1]
# finetune_thetas = finetune_thetas_all[:, -1, :]
# print(f"Shape of finetune_thetas: {finetune_thetas_all.shape}")





#%% Let's finetune theta0 for 256 gradient descent steps on the trajectory
def tinetune_loss_fn(thet):
    X_recons = apply_theta(thet)
    loss_r = optax.l2_loss(X_recons, xs_true[0])
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
# min_val = 0.
# max_val = 1.
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

#%% Plot the last matrix 

final_res = results_matrix[:, -1, :]

## final res as the diagonal of the matrix
final_res = np.zeros((256, 2))
for i in range(256):
    final_res[i, 0] = results_matrix[i, i, 0]
    final_res[i, 1] = results_matrix[i, i, 1]

fig, axs = plt.subplots(1, 1, figsize=(10, 5))

plt.plot(final_res[:, 0], 'b-', lw=2, label="X")
plt.plot(final_res[:, 1], 'g-', lw=2, label="Y")

# plt.title("Final Theta")
plt.xlabel(r"$t$")
# plt.ylabel("Value")

plt.legend()
plt.draw();




# #%% Lets analyse eigen values of the matrix A

# A = model.As[0]
# print(f"Shape of A: {A.shape}")
# eigs = np.linalg.eigvals(A)


# ## Check if the eigen values are complex
# complex_eigs = np.iscomplex(eigs)
# print(f"Number of complex eigen values: {np.sum(complex_eigs)}")
# print(f"Number of real eigen values: {np.sum(~complex_eigs)}")
# ## Print the real and imaginary parts of the complex eigen values
# print(f"Real part of complex eigen values: {np.real(eigs[complex_eigs])}")
# print(f"Imaginary part of complex eigen values: {np.imag(eigs[complex_eigs])}")

# ## Plot the real against the  complex parts of the eigen values 
# # real_eigs = np.real(eigs[~complex_eigs])
# real_part = np.real(eigs[complex_eigs])
# complex_part = np.imag(eigs[complex_eigs])

# fig, axs = plt.subplots(1, 1, figsize=(10, 10))
# plt.plot(real_part, complex_part, 'bo', lw=2)
# plt.title("Eigen values of A")
# plt.xlabel("Real part")
# plt.ylabel("Imaginary part")
# plt.draw();


# ## Check whether the matrix is stable
# stable = np.all(np.abs(eigs) < 1)
# print(f"Matrix is stable: {stable}")
# if stable:
#     print("Matrix is stable")
# else:
#     print("Matrix is not stable")
#     print(f"Eigen values: {eigs}")
#     print(f"Max eigen value: {np.max(np.abs(eigs))}")
#     print(f"Min eigen value: {np.min(np.abs(eigs))}")
#     print(f"Mean eigen value: {np.mean(np.abs(eigs))}")
#     print(f"Median eigen value: {np.median(np.abs(eigs))}")
#     print(f"Std eigen value: {np.std(np.abs(eigs))}")



#%% Lets check the correlation between the weights (256, 4098) and time (256, 1). 

@eqx.filter_vmap
@eqx.filter_vmap(in_axes=(1))
def correlation(a):
    """ Calculate the correlation between two arrays. """
    b = jnp.linspace(0, 1, a.shape[0])
    a = a - jnp.mean(a)
    b = b - jnp.mean(b)
    return jnp.sum(a*b) / (jnp.sqrt(jnp.sum(a**2)) * jnp.sqrt(jnp.sum(b**2)))

# corr_coeffs = eqx.filter_vmap(correlation)(thetas[0, :, :].T)
corr_coeffs = correlation(thetas[:5, :, :])
print(f"Shape of correlation coeffs: {corr_coeffs.shape}")
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(corr_coeffs.T, '.', lw=2, alpha=0.5, markersize=5)
ax.set_title("Correlation between weights and time")
ax.set_xlabel("Weight Dimension")
ax.set_ylabel("Correlation Coefficient")
plt.draw();
plt.savefig(plots_folder+"correlation.png", dpi=100, bbox_inches='tight')


## Plot the correlation coeffs as a histogram
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

## Histogram with normalisation
ax.hist(corr_coeffs.flatten(), bins=100, color='purple', alpha=0.75, density=True)

# ax.set_title("Correlation Coefficients Histogram")
ax.set_xlabel("Correlation Coefficient")
ax.set_ylabel("Density")
plt.draw();
plt.savefig(plots_folder+"correlation_hist.pdf", dpi=100, bbox_inches='tight')

#%% Visualise the correlation between the weights and time
#%%[markdown]

# ## Recurrent Neural Networks

#%% Import the necessary modules

# %load_ext autoreload
# %autoreload 2

from utils import *
from loaders import *
from models import *

import jax
print("\n\nAvailable devices:", jax.devices())

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








#%% # Parse the command line arguments or use the default config.yaml file

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

model_type = config['model']['model_type']
if model_type == "wsm":
    print("\n\n+=+=+=+=+ Training Weight Space Model +=+=+=+=+\n")
elif model_type == "gru":
    print("\n\n+=+=+=+=+ Training Gated Recurrent Unit Model +=+=+=+=+\n")
elif model_type == "lstm":
    print("\n\n+=+=+=+=+ Training Long Short Term Memory Model +=+=+=+=+\n")
elif model_type == "ffnn":
    print("\n\n+=+=+=+=+ Training Feed-Forward Neural Network Model +=+=+=+=+\n")
else:
    print("\n\n+=+=+=+=+ Training Unknown Model +=+=+=+=+\n")
    raise ValueError(f"Unknown model type: {model_type}")

seed = config['general']['seed']
main_key = jax.random.PRNGKey(seed)
np.random.seed(seed)
torch.manual_seed(seed)






#%% Setup the run folder and data folder
train = config['general']['train']
classification = config['general']['classification']
data_path = config['general']['data_path']
save_path = config['general']['save_path']

### Create and setup the run and data folders
if train:
    if save_path is not None:
        run_folder = save_path
    else:
        run_folder = make_run_folder(f'./runs/{config["general"]["dataset"]}/')
    data_folder = data_path
else:
    run_folder = "./"
    data_folder = f"../../../{data_path}"

print("Using run folder:", run_folder)
## Copy the module files to the run folder
logger, checkpoints_folder, plots_folder, artefacts_folder = setup_run_folder(run_folder, training=train)

## Copy the config file to the run folder, renaming it as config.yaml
if not os.path.exists(run_folder+"config.yaml"):
    test_config = config.copy()
    test_config['general']['train'] = False             ## Set the train flag to False
    with open(run_folder+"config.yaml", 'w') as file:
        yaml.dump(test_config, file)

## Print the config file using the logger
logger.info(f"Config file: {args.config_file}")
logger.info("==== Config file's contents ====")

for key, value in config.items():
    if isinstance(value, dict):
        logger.info(f"{key}:")
        for sub_key, sub_value in value.items():
            logger.info(f"  {sub_key}: {sub_value}")
    else:
        logger.info(f"{key}: {value}")





#%% Create the data loaders and visualize a few samples

trainloader, validloader, testloader, data_props = make_dataloaders(data_folder, config)
nb_classes, seq_length, data_size, width = data_props

print("Total number training samples:", len(trainloader.dataset))

batch = next(iter(trainloader))
(in_sequence, times), output = batch
logger.info(f"Input sequence shape: {in_sequence.shape}")
logger.info(f"Labels/Output Sequence shape: {output.shape}")
logger.info(f"Seq length: {seq_length}")
logger.info(f"Data size: {data_size}")
logger.info(f"Min/Max in the dataset: {np.min(in_sequence), np.max(in_sequence)}")
logger.info("Number of batches:")
logger.info(f"  - Train: {trainloader.num_batches}")
logger.info(f"  - Valid: {validloader.num_batches}")
logger.info(f"  - Test: {testloader.num_batches}")

## Plot a few samples in a 4x4 grid (chose them at random)
fig, axs = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
colors = ['r', 'g', 'b', 'c', 'm', 'y']

dataset = config['general']['dataset']
image_datasets = ["mnist", "mnist_fashion", "cifar", "celeba", "pathfinder"]
dynamics_datasets = ["lorentz63", "lorentz96", "lotka", "trends", "mass_spring_damper", "cheetah", "electricity", "sine"]
repeat_datasets = ["lotka", "arc_agi", "icl", "traffic", "mitsui"]

res = (width, width, data_size)
dim0, dim1 = (0, -1)
if dim1>= data_size:
    dim1 = 0
    logger.info(f"dim1 is out of bounds. Setting it to 0.")

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
            ## Make 4 plots against each other dimensions
            # axs[i, j].plot(in_sequence[idx, :, dim0], in_sequence[idx, :, dim1], color=colors[(i*j)%len(colors)])
            # axs[i, j].plot(output[idx, :, dim0], output[idx, :, dim1], color=colors[(i*j)%len(colors)], linestyle='--')
            ## Make 4 plots against time steps
            axs[i, j].plot(output[idx, :, dim0], color=colors[(i*j)%len(colors)], linestyle='-', lw=1, alpha=0.5)
            axs[i, j].plot(output[idx, :, dim1], color=colors[(i*j)%len(colors)], linestyle='--', lw=1, alpha=0.5)
            axs[i, j].plot(in_sequence[idx, :, dim0], color=colors[(i*j)%len(colors)], lw=3)
            axs[i, j].plot(in_sequence[idx, :, dim1], color=colors[(i*j)%len(colors)], linestyle='--', lw=3)
        else:
            # axs[i, j].plot(in_sequence[idx, :, dim0], in_sequence[idx, :, dim1], color=colors[output[idx]%len(colors)])
            axs[i, j].plot(in_sequence[idx, :, dim0], color=colors[int(output[idx])%len(colors)], lw=3)
            axs[i, j].plot(in_sequence[idx, :, dim1], color=colors[int(output[idx])%len(colors)], linestyle='--', lw=3)

        if dataset not in repeat_datasets:
            axs[i, j].set_title(f"Class: {output[idx]}", fontsize=12)
        axs[i, j].axis('off')

plt.suptitle(f"{dataset.upper()} Training Samples", fontsize=20)
plt.draw();
plt.savefig(plots_folder+"samples_train.png", dpi=100, bbox_inches='tight')



# %% Define the model and loss function

model_key, train_key, test_key = jax.random.split(main_key, num=3)
if not classification:
    nb_classes = None
model = make_model(model_key, data_size, nb_classes, config, logger)
untrained_model = model

nb_recons_loss_steps = config['training']['nb_recons_loss_steps']
use_nll_loss = config['training']['use_nll_loss']

def loss_fn(model, batch, key):
    """ Loss function for the model. A batch contains: (Xs, Ts), Ys
    Xs: (batch, time, data_size)
    Ts: (batch, time)
    Ys: (batch, num_classes)
    """

    if not classification:  ## Regression (forecasting) task
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
            # if dataset == "icl":        ## We only care about the last column, last row
            #     print("\n\n     Considering the last column, last row of the ICL dataset for the loss ...")
            #     # print("     X_recons_ shape:", X_recons_.shape, "\n\n")
            #     # loss_r = optax.l2_loss(X_recons_[:, -1,-1], X_true_[:, -1,-1])
            #     # loss_r = optax.l2_loss(X_recons_[:, :, -1], X_true_[:, :, -1])

            #     ## Using key, randomly select between the last and one before last time step
            #     inside = jax.random.randint(key, (1,), 0, X_recons_.shape[1]-1)[0]
            #     index = jnp.where(jax.random.uniform(key, (1,)) < 0.5, inside, -1)
            #     loss_r = optax.l2_loss(X_recons_[:, index, -1], X_true_[:, index, -1])
            # else:
            #     loss_r = optax.l2_loss(X_recons_, X_true_)

            loss_r = optax.l2_loss(X_recons_, X_true_)

        loss = jnp.mean(loss_r)
        return loss, (loss,)

    else:           ## Classification task
        (X_true, times), Ys = batch

        Y_hat = model(X_true, times, key, inference_start=None)

        # Compute the cross entropy loss using Optax
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=Y_hat[:, -1], labels=Ys)

        loss = jnp.mean(loss)

        acc = jnp.mean(jnp.argmax(Y_hat[:, -1], axis=-1) == Ys)

        return loss, (acc,)


@eqx.filter_jit
def train_step(model, batch, opt_state, key):
    print('\nCompiling function "train_step" ...')

    (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, key)

    updates, opt_state = opt.update(grads, opt_state, model, value=loss)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, aux_data


@eqx.filter_jit
def forward_pass(model, X, times, key, inference_start=None):
    """ Jitted forward pass. """
    X_recons = model(X, times, key, inference_start)
    return X_recons


val_criterion = config['training']['val_criterion']
if val_criterion == "nll":
    if not config['training']['stochastic']:
        raise ValueError("NLL val loss can only be used if trained in stochastic mode.")
    elif "smooth_inference" in config['training'] and config['training']['smooth_inference']:
        raise ValueError("NLL val loss cannot be used in smooth inference mode (since only the mean is predicted).")

logger.info(f"Validation criterion is: {val_criterion}")



def eval_on_dataloader(model, dataloader, inference_start, key):
    """ Evaluate the model on the designated set. """

    cri_vals = []
    new_key, _ = jax.random.split(key)

    for i, batch in enumerate(dataloader):
        new_key, _ = jax.random.split(new_key)
        (X_true, times), X_labs_outs = batch

        if not classification:
            X_recons = forward_pass(model, X_true, times, new_key, inference_start=inference_start)

            if dataset in repeat_datasets:
                X_gt = X_labs_outs
            else:
                X_gt = X_true

            if use_nll_loss:
                means_, stds_ = jnp.split(X_recons, 2, axis=-1)
            else:
                means_ = X_recons

            if val_criterion == "mse":
                loss_val = optax.l2_loss(means_, X_gt).mean()
            elif val_criterion == "mae":
                # loss_val = optax.l1_loss(means_, X_gt).mean()       ## TODO: not implemented yet
                loss_val = jnp.mean(jnp.abs(means_ - X_gt))
            elif val_criterion == "rmse":
                # loss_val = optax.root_mean_squared_error(means_, X_gt).mean()   ## TODO: not implemented yet
                loss_val = jnp.sqrt(jnp.mean((means_ - X_gt)**2))
            elif val_criterion == "nll":
                loss_val = jnp.log(stds_) + 0.5*((X_gt - means_)/stds_)**2
                loss_val = jnp.mean(loss_val)
            else:
                raise ValueError(f"Unknown validation criterion for regression: {val_criterion}")

        else:
            Y_hat = forward_pass(model, X_true, times, new_key, inference_start=inference_start)
            if val_criterion == "cce":
                ## ==== Use the categorical cross-entropy loss for validation ====
                loss_val = optax.softmax_cross_entropy_with_integer_labels(logits=Y_hat[:, -1], labels=X_labs_outs).mean()
            elif val_criterion == "error_rate":
                ## ==== Let "cce" be the complement accuracy instead (the lower the better) ===
                loss_val = jnp.mean(jnp.argmax(Y_hat[:, -1], axis=-1) == X_labs_outs)
                loss_val = 1 - loss_val
            elif val_criterion == "f1_score": ## F1_score, but MACRO, as below (but with JAX)
            # from torchmetrics import F1Score
            # f1_macro = F1Score(task="multiclass", average="macro", num_classes=len(disp_labels)).to(module.device)
                preds = jnp.argmax(Y_hat[:, -1], axis=-1)
                f1_macro = f1_score_macro(y_true=X_labs_outs, y_pred=preds, nb_classes=nb_classes)
                loss_val = 1 - f1_macro
            else:
                raise ValueError(f"Unknown validation criterion for classification: {val_criterion}")

        cri_vals.append(loss_val)

    return np.mean(cri_vals), np.median(cri_vals), np.min(cri_vals)






#%% Train and validate the model

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

    val_losses = []
    best_val_loss = np.inf
    best_val_loss_epoch = 0

    print_every = config['training']['print_every']
    save_every = config['training']['save_every']
    valid_every = config['training']['valid_every']
    inf_start = config["training"]["inference_start"]

    nb_epochs = config['training']['nb_epochs']
    logger.info(f"\n\n=== Beginning training ... ===")
    logger.info(f"  - Number of epochs: {nb_epochs}")
    logger.info(f"  - Number of batches: {trainloader.num_batches}")
    logger.info(f"  - Total number of GD steps: {trainloader.num_batches*nb_epochs}")

    start_time = time.time()

    for epoch in range(nb_epochs):

        epoch_start_time = time.time()
        losses_epoch = []
        aux_epoch = []

        for i, batch in enumerate(trainloader):
            train_key, _ = jax.random.split(train_key)
            model, opt_state, loss, aux = train_step(model, batch, opt_state, train_key)

            losses_epoch.append(loss)
            losses.append(loss)
            aux_epoch.append(aux)

            lr_scales.append(optax.tree_utils.tree_get(opt_state, "scale"))

        mean_epoch, median_epoch = np.mean(losses_epoch), np.median(losses_epoch)
        epoch_end_time = time.time() - epoch_start_time

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            logger.info(
                f"Epoch {epoch:-4d}/{nb_epochs:-4d}     Train Loss   -Mean: {mean_epoch:.6f},   -Median: {median_epoch:.6f},   -Latest: {loss:.6f},     -WallTime: {epoch_end_time:.2f} secs"
            )

            if classification:
                logger.info(f"Average train classification accuracy: {np.mean(aux_epoch)*100:.2f}%")

        if epoch%save_every==0 or epoch==nb_epochs-1:
            if epoch==nb_epochs-1:  ## @TODO: to save space !
                eqx.tree_serialise_leaves(checkpoints_folder+f"model_{epoch}.eqx", model)
            np.save(artefacts_folder+"losses.npy", np.array(losses))
            np.save(artefacts_folder+"lr_scales.npy", np.array(lr_scales))
            np.savez(artefacts_folder+"val_losses.npz", losses=np.array(val_losses), best_epoch=best_val_loss_epoch, best_loss=best_val_loss)

            ## Only save the best model with the lowest mean loss
            med_losses_per_epoch.append(median_epoch)
            if epoch==0 or median_epoch<=np.min(med_losses_per_epoch[:-1]):
                eqx.tree_serialise_leaves(artefacts_folder+"model_train.eqx", model)
                with open(artefacts_folder+"opt_state.pkl", 'wb') as f:
                    pickle.dump(opt_state, f)
                logger.info("Best model on training set saved ...")

        if (valid_every is not None) and (epoch%valid_every==0) or (epoch==nb_epochs-1):
            val_mean_loss, val_median_loss, _ = eval_on_dataloader(model, validloader, inference_start=inf_start, key=test_key)
            val_losses.append(val_mean_loss)

            logger.info(
                f"Epoch {epoch:-4d}/{nb_epochs:-4d}     Validation Loss   +Mean: {val_mean_loss:.6f},   +Median: {val_median_loss:.6f}"
            )

            ## Save the model with the lowest validation loss
            if epoch==0 or val_mean_loss<=np.min(val_losses[:-1]):
                eqx.tree_serialise_leaves(artefacts_folder+"model.eqx", model)
                logger.info("Best model on validation set saved ...")
                best_val_loss = val_mean_loss
                best_val_loss_epoch = epoch

        if epoch==3:     ## Print the output of nvidia-smi to check VRAM usage
            os.system("nvidia-smi")
            os.system("nvidia-smi >> "+artefacts_folder+"training.log") 

    wall_time = time.time() - start_time
    logger.info("\nTraining complete. Total time: %d hours %d mins %d secs" %seconds_to_hours(wall_time))

    ## Restore the best model
    if os.path.exists(artefacts_folder+"model.eqx"):
        model = eqx.tree_deserialise_leaves(artefacts_folder+"model.eqx", model)
        logger.info(f"Best model from epoch {best_val_loss_epoch} restored.")
    elif os.path.exists(artefacts_folder+"model_train.eqx"):
        model = eqx.tree_deserialise_leaves(artefacts_folder+"model_train.eqx", model)
        logger.info(f"Best model on 'training set' restored.")
    else:
        logger.info("No 'best' model found. Using the last model.")



else:
    if os.path.exists(artefacts_folder+"model.eqx"):
        model = eqx.tree_deserialise_leaves(artefacts_folder+"model.eqx", model)
        logger.info(f"Best validation model restored.")
    elif os.path.exists(artefacts_folder+"model_train.eqx"):
        model = eqx.tree_deserialise_leaves(artefacts_folder+"model_train.eqx", model)
        logger.info(f"Best model on 'training set' restored.")
    else:
        raise ValueError("No model found to load. You might want to use one from a checkpoint.")

    try:
        losses = np.load(artefacts_folder+"losses.npy")
        lr_scales = np.load(artefacts_folder+"lr_scales.npy")
        val_losses_raw = np.load(artefacts_folder+"val_losses.npz")
        val_losses = val_losses_raw['losses']
        best_val_loss_epoch = val_losses_raw['best_epoch'].item()
        best_val_loss = val_losses_raw['best_loss'].item()
    except:
        losses = []
        val_losses = []

    logger.info(f"Model loaded from {run_folder}model.eqx")







# %% Visualise the training (and validation) losses

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
train_steps = np.arange(len(losses))
loss_name = "NLL" if use_nll_loss else r"$L_2$"
ax = sbplot(train_steps, clean_losses, color="purple", title="Loss History", x_label='Train Steps', y_label=loss_name, ax=ax, y_scale="linear" if use_nll_loss else "log", label="Train");
ax.legend(fontsize=16, loc='upper left')
ax.yaxis.label.set_color('purple')

## Make a twin axis for the validation losses
nb_epochs = config['training']['nb_epochs']
valid_every = config['training']['valid_every']
if len(val_losses) > 0:
    val_col = "teal"
    ax_ = ax.twinx()
    epochs_ids = (np.arange(0, nb_epochs, valid_every).tolist() + [nb_epochs-1])[:len(val_losses)]
    val_steps_ids = (np.array(epochs_ids)+1) * trainloader.num_batches      ## Convert epochs to train steps
    ax_ = sbplot(val_steps_ids, val_losses, ".-", color=val_col, label=f"Valid", y_label=f'{val_criterion.upper()}', ax=ax_, y_scale="linear" if val_criterion in ["nll", "cce", "error_rate"] else "log", linewidth=3);
    ax_.legend(fontsize=16, loc='upper right')
    ax_.yaxis.label.set_color(val_col)

clean_losses = np.where(clean_losses<np.percentile(clean_losses, 96), clean_losses, np.nan)
## Plot a second training loss plot with the outliers removed
ax2 = sbplot(train_steps, clean_losses, title="Loss History (96th Percentile)", x_label='Train Steps', y_label=loss_name, ax=ax2, y_scale="linear" if use_nll_loss else "log");

plt.draw();
plt.savefig(plots_folder+"loss.png", dpi=100, bbox_inches='tight')

if os.path.exists(artefacts_folder+"lr_scales.npy"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax = sbplot(lr_scales, "g-", title="LR Scales", x_label='Train Steps', ax=ax, y_scale="log");

    # plt.legend()
    plt.draw();
    plt.savefig(plots_folder+"lr_scales.png", dpi=100, bbox_inches='tight')



## Plot the validation losses (in more details this time)
nb_epochs = config['training']['nb_epochs']
valid_every = config['training']['valid_every']
val_ids = (np.arange(0, nb_epochs, valid_every).tolist() + [nb_epochs-1])[:len(val_losses)]
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax = sbplot(val_ids, val_losses, title=f"{val_criterion.upper()} on Valid Set at Various Epochs", x_label='Epoch', y_label=f'{val_criterion}', ax=ax, y_scale="log", linewidth=3);
plt.axvline(x=best_val_loss_epoch, color='r', linestyle='--', linewidth=3, label=f"Best {val_criterion.upper()}: {best_val_loss:.6f} at Epoch {best_val_loss_epoch}")
plt.legend(fontsize=16)
plt.draw();
plt.savefig(plots_folder+f"checkpoints_{val_criterion.lower()}.png", dpi=100, bbox_inches='tight')
logger.info(f"Best model found at epoch {best_val_loss_epoch} with {val_criterion}: {best_val_loss:.6f}")




# %% Other visualisations of the model

if config["model"]["model_type"] == "wsm":

    ## Let's visualise the distribution of values along the main diagonal of A and theta
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].hist(jnp.diag(model.As[0], k=0), bins=100)

    axs[0].set_title("Histogram of diagonal values of A")

    if hasattr(model, "thetas"):
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
    min_val = -0.0000
    max_val = 0.00003

    img = axs[0].imshow(untrained_model.As[0], cmap='viridis', vmin=min_val, vmax=max_val)
    axs[0].set_title("Untrained A")
    plt.colorbar(img, ax=axs[0], shrink=0.7)

    img = axs[1].imshow(model.As[0], cmap='viridis', vmin=min_val, vmax=max_val)
    axs[1].set_title("Trained A")
    plt.colorbar(img, ax=axs[1], shrink=0.7)
    plt.draw();
    plt.savefig(plots_folder+"A_matrices.png", dpi=100, bbox_inches='tight')

    ## Visualize the dynamic tanh attributes (if they exist)
    if isinstance(config['model']['root_final_activation'], list) and not classification:
        latex_string = r"y = \alpha \cdot \text{tanh} \left( \frac{x-b}{a} \right) + \beta"
        logger.info(f"Dynamic tanh params (final root network activation) : ${latex_string}$ ")

        display(Math(latex_string))
        logger.info(f"a, b, alpha, beta: {model.dtanh_params}")

        ## Plot this against a normal tanh
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        x = np.linspace(-35, 35, 500)
        y = np.tanh(x)
        a, b, alpha, beta = model.dtanh_params
        y2 = alpha * np.tanh((x-b)/a) + beta
        ax.plot(x, y, label="tanh")
        ax.plot(x, y2, label="dynamic tanh")
        ax.set_title("Dynamic tanh vs tanh after training")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        plt.draw();
        plt.savefig(plots_folder+"dynamic_tanh.png", dpi=100, bbox_inches='tight')


# %% Visualising a few reconstruction samples (for regression tasks)

if not classification:
    visloader = NumpyLoader(testloader.dataset, batch_size=16, shuffle=True)
    nb_examples = len(visloader.dataset)    ## Actual number of examples in the dataset

    nb_cols = 3 if use_nll_loss else 2
    fig, axs = plt.subplots(4, 4*nb_cols, figsize=(16*3, 16), sharex=True, constrained_layout=True)

    batch = next(iter(visloader))
    (xs_ins, times), xs_outs_labs = batch
    xs_true = xs_outs_labs if dataset in repeat_datasets else xs_ins

    inference_start = config['training']['inference_start']
    xs_recons = forward_pass(model=model, 
                        X=xs_ins, 
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
            x = xs_true[(i*4+j)%nb_examples]
            x_recons = xs_recons[(i*4+j)%nb_examples]
            x_full = xs_true[(i*4+j)%nb_examples]

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
                axs[i, nb_cols*j].set_ylim([min(min_0, min_1)-eps, max(max_0, max_1)+eps])
                axs[i, nb_cols*j].plot(x_full[:, dim0], color=colors[(i*4+j)%len(colors)])
                axs[i, nb_cols*j].plot(x_full[:, dim1], color=colors[(i*4+j)%len(colors)], linestyle='-.')
            else:
                axs[i, nb_cols*j].set_ylim([min(min_0, min_1)-eps, max(max_0, max_1)+eps])
                axs[i, nb_cols*j].plot(x[:, dim0], color=colors[int(labels[(i*4+j)%nb_examples])%len(colors)])
                axs[i, nb_cols*j].plot(x[:, dim1], color=colors[int(labels[(i*4+j)%nb_examples])%len(colors)], linestyle='-.')
            if i==0:
                axs[i, nb_cols*j].set_title("GT", fontsize=40)
            # axs[i, nb_cols*j].axis('off')

            if dataset in image_datasets:
                to_plot = x_recons.reshape(res)
                if dataset=="celeba":
                    to_plot = (to_plot + 1) / 2
                axs[i, nb_cols*j+1].imshow(to_plot, cmap='gray')
            elif dataset in repeat_datasets:
                axs[i, nb_cols*j+1].set_ylim([min(min_0, min_1)-eps, max(max_0, max_1)+eps])
                axs[i, nb_cols*j+1].plot(x_recons[:, dim0], color=colors[(i*4+j)%len(colors)])
                axs[i, nb_cols*j+1].plot(x_recons[:, dim1], color=colors[(i*4+j)%len(colors)], linestyle='-.')
            else:
                axs[i, nb_cols*j+1].set_ylim([min(min_0, min_1)-eps, max(max_0, max_1)+eps])
                axs[i, nb_cols*j+1].plot(x_recons[:, dim0], color=colors[labels[(i*4+j)%nb_examples]%len(colors)])
                axs[i, nb_cols*j+1].plot(x_recons[:, dim1], color=colors[labels[(i*4+j)%nb_examples]%len(colors)], linestyle='-.')

            if i==0:
                axs[i, nb_cols*j+1].set_title("Recons", fontsize=40)
            # axs[i, nb_cols*j+1].axis('off')

            if use_nll_loss and dataset not in repeat_datasets:
                logger.info(f"Min/Max Uncertainty: {np.min(xs_uncert):.3f}, {np.max(xs_uncert):.3f}")
                if dataset in image_datasets:
                    to_plot = xs_uncert[i*4+j].reshape(res)
                    axs[i, nb_cols*j+2].imshow(to_plot, cmap='gray')
                else:
                    to_plot = xs_uncert[i*4+j]
                    axs[i, nb_cols*j+2].plot(to_plot[:, dim0], color=colors[labels[i*4+j]%len(colors)])
                    axs[i, nb_cols*j+2].plot(to_plot[:, dim1], color=colors[labels[i*4+j]%len(colors)], linestyle='-.')

                if i==0:
                    axs[i, nb_cols*j+2].set_title("Uncertainty", fontsize=36)
                # axs[i, nb_cols*j+2].axis('off')

    plt.suptitle(f"Reconstruction using {inference_start} initial steps", fontsize=65)
    plt.draw();
    plt.savefig(plots_folder+"samples_generated.png", dpi=100, bbox_inches='tight')
    # plt.savefig(plots_folder+"samples_generated.pdf", dpi=300, bbox_inches='tight')



#%% Evaluate the model on the entire test set (for regression tasks)

if not classification:
    eval_loader = NumpyLoader(testloader.dataset, batch_size=len(testloader.dataset), shuffle=False)

    batch = next(iter(eval_loader))
    (xs_ins, times), xs_out_labs = batch
    xs_true = xs_out_labs if dataset in repeat_datasets else xs_ins

    print("Inferance starts at: ", inference_start)
    xs_recons = forward_pass(model=model, 
                        X=xs_ins, 
                        times=times, 
                        key=test_key, 
                        inference_start=inference_start)

    xs_recons = xs_recons[:, :, :data_size]

    def metrics(pred, true):
        """ Calculate the metrics (after inference starts) for the predictions and the true values."""
        pred = pred[:, inference_start:, :]
        true = true[:, inference_start:, :]

        mse = jnp.mean((pred - true)**2)
        rmse = jnp.sqrt(mse)
        mae = jnp.mean(jnp.abs(pred - true))
        mape = jnp.mean(jnp.abs(pred - true)/jnp.abs(true))
        mspe = jnp.mean(jnp.abs(pred - true)/jnp.abs(true)**2)
        return mse, rmse, mae, mape, mspe

    mse, rmse, mae, mape, mspe = metrics(xs_recons, xs_true)
    logger.info("Evaluation of forecast MSE on the full test set in inference mode:")
    logger.info(f"    - MSE : {mse:.6f}")
    logger.info(f"    - RMSE : {rmse:.6f}")
    logger.info(f"    - MAE : {mae:.6f}")
    logger.info(f"    - MAPE : {mape:.6f}")


## If possible, unormalize the test data and the predictions before computing the metrics
if not classification and config['data']['normalize']:
    def unormalize_data(y, min_max):
        """ Unnormalize the data using the min and max values of the dataset. """

        ## We assume y was obtained as
        # y = (x - min_max[0]) / (min_max[1] - min_max[0])
        # y = (y - 0.5) * 2

        ## We want to get back to the original data
        x = (y + 1) / 2
        x = x * (min_max[1] - min_max[0]) + min_max[0]

        return x

    if hasattr(testloader.dataset, "min_data") and hasattr(testloader.dataset, "max_data"):
        min_max = (trainloader.dataset.min_data, trainloader.dataset.max_data)

        if dataset != "mitsui":
            xs_recons_unorm = unormalize_data(xs_recons, min_max)
            xs_true_unorm = unormalize_data(xs_true, min_max)
        else:
            xs_recons_unorm = xs_recons
            xs_true_unorm = xs_true

        ## Plot a few samples of the unormalized data
        plot_dim = np.random.randint(0, xs_recons_unorm.shape[-1], size=2)[0]
        plot_id = np.random.randint(0, xs_true_unorm.shape[0], size=1)[0]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
        ax.plot(xs_true_unorm[plot_id, :, plot_dim], "r-", lw=1, label="True")
        ax.plot(xs_recons_unorm[plot_id, :, plot_dim], "r-", lw=3, label="Recons")
        ax.set_title(f"Unormalised - Dim {plot_dim}")

        ## Plot on ax2 the normalized
        ax2.plot(xs_true[plot_id, :, plot_dim], "b-", lw=1, label="True")
        ax2.plot(xs_recons[plot_id, :, plot_dim], "b-", lw=3, label="Recons")
        ax2.set_title(f"Normalised - Dim {plot_dim}")

        ax.legend()
        # ax2.legend()

        plt.tight_layout()

        mse, rmse, mae, mape, mspe = metrics(xs_recons_unorm, xs_true_unorm)
        logger.info("Evaluation of forecast MSE on the full test set in inference mode - UNORMALISED DATA:")
        logger.info(f"    - MSE : {mse:.6f}")
        logger.info(f"    - RMSE : {rmse:.6f}")
        logger.info(f"    - MAE : {mae:.6f}")
        logger.info(f"    - MAPE : {mape:.6f}")















# %% Evaluate the model on the test set (for classification tasks)

if classification:
    evalloader = NumpyLoader(testloader.dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    accs = []
    f1s = []
    for i, batch in enumerate(evalloader):
        (in_sequence, times), output = batch
        Y_hat_raw = forward_pass(model, in_sequence, times, main_key)
        Y_hat = jnp.argmax(Y_hat_raw, axis=-1)
        acc = jnp.mean(Y_hat[:, -1] == output)
        f1 = f1_score_macro(y_true=output, y_pred=Y_hat[:, -1], nb_classes=nb_classes)

        accs.append(acc)
        f1s.append(f1)

    accuracy = np.mean(accs)
    logger.info(f"Test set accuracy: {accuracy:.4f}")
    f1_macro = np.mean(f1s)
    logger.info(f"Test set F1-macro: {f1_macro:.4f}")

    ## Visualise a few model logits (the first 16 sequences)
    visloader = NumpyLoader(testloader.dataset, batch_size=16, shuffle=True)
    batch = next(iter(visloader))
    (in_sequence, times), output = batch

    fig, axs = plt.subplots(4, 4, figsize=(20, 15), sharex=True)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for i in range(4):
        for j in range(4):
            idx = i*4+j

            if dataset in image_datasets:
                to_plot = in_sequence[idx].reshape(res)
                axs[i, j].imshow(to_plot, cmap='gray')

            else:
                if dataset=="spirals":
                    axs[i, j].plot(in_sequence[idx, :, dim0], in_sequence[idx, :, dim1], color=colors[int(output[idx])%len(colors)], lw=3)
                else:
                    axs[i, j].plot(in_sequence[idx, :], color=colors[int(output[idx])%len(colors)], lw=3)

            axs[i, j].set_title(f"Predicted Class: {Y_hat[idx, -1]}", fontsize=12)
            axs[i, j].axis('off')

    plt.suptitle(f"{dataset.upper()} Test Samples", fontsize=20)

    plt.draw();
    plt.savefig(plots_folder+"samples_test.png", dpi=100, bbox_inches='tight')


    ## PLot the first 16 seqences class over time, with the title as the predicted class
    fig, axs = plt.subplots(4, 4, figsize=(20, 15), sharex=True, sharey=True)

    for i in range(4):
        for j in range(4):
            idx = i*4+j

            axs[i, j].plot(Y_hat[idx, :], color=colors[int(output[idx])%len(colors)], lw=3)

            axs[i, j].set_title(f"Predicted Class: {Y_hat[idx, -1]}", fontsize=22)
            # axs[i, j].axis('off')

            axs[i, j].set_ylim([-0.1, nb_classes-1+0.1])

            axs[i, j].set_yticks(np.arange(nb_classes))
            axs[i, j].set_yticklabels(np.arange(nb_classes))

            if i==3:
                axs[i, j].set_xlabel("Time Step", fontsize=22)

    plt.tight_layout()
    plt.suptitle(f"{dataset.upper()} Predicted Test Labels", fontsize=20, y=1.02)

    plt.draw();
    plt.savefig(plots_folder+"samples_test_labels.png", dpi=100, bbox_inches='tight')


# %% Special visualisation for ICL dataset with 1D x and 1D y. PLot a scatter plot, true in green, recons in red (using y_hat)

if dataset == "icl":
    eval_loader = NumpyLoader(testloader.dataset, batch_size=len(testloader.dataset), shuffle=False)

    batch = next(iter(eval_loader))
    (xs_true, times), ys_true = batch

    inference_start = config['training']['inference_start']
    xs_recons = forward_pass(model=model, 
                        X=xs_true, 
                        times=times, 
                        key=test_key, 
                        inference_start=inference_start)

    xs_recons = xs_recons[:, :, :data_size]

    ## Make the data to plot
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown']
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # for batch_el in range(0, xs_true.shape[0]):
    for el in range(0, 6):
        batch_el = np.random.randint(0, xs_true.shape[0], size=1)[0]  # Randomly select a batch element
        Xs = xs_true[batch_el, :, 0]  # All time steps, first dimension
        Ys = ys_true[batch_el, :, -1]  # All time steps, last dimension
        Ys_hat = xs_recons[batch_el, :, -1]

        ## Order the Xs for better visualisation
        order = np.argsort(Xs)
        Xs = Xs[order]
        Ys = Ys[order]
        Ys_hat = Ys_hat[order]

        ## Plot the true with dots, and predicted with + markers.
        color = colors[el % len(colors)]
        alpha = np.random.uniform(0.75, 1.0)
        ax.scatter(Xs, Ys, facecolors='none', edgecolors=color, label='True' if el==0 else None, s=25, alpha=0.5)
        ax.scatter(Xs, Ys_hat, color=color, marker='+', label='Pred' if el==0 else None, s=45, alpha=1)

    ax.set_xlabel(r"$\mathbf{x}_0$", fontsize=40)
    ax.set_ylabel(r"$y, \hat{y}$", fontsize=40)
    ax.set_title("ICL Dataset's Keys and Query Points")
    ax.legend(fontsize=30)
    plt.draw();
    plt.savefig(plots_folder+"icl_all_keys_plus_query.png", dpi=100, bbox_inches='tight')      

# %% # %% Special visualisation for ICL dataset. But this time, we only consider the query points
if dataset == "icl":
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    Xs = xs_true[:, -1, 0]  # All time steps, first dimension
    Ys = ys_true[:, -1, -1]  # All time steps, last dimension
    Ys_hat = xs_recons[:, -1, -1]

    ax.scatter(Ys, Ys_hat, color="crimson", marker='X', s=80, alpha=0.5)

    ax.set_xlabel(r"$y_q$", fontsize=40)
    ax.set_ylabel(r"$\hat{y}_q$", fontsize=40)
    ax.set_title("ICL Dataset's Queries Only")

    ## Add a diagonal line
    ax.plot([np.min(Ys), np.max(Ys)], [np.min(Ys), np.max(Ys)], color='black', linestyle='--', linewidth=2, label="$y_q = \hat{y}_q$")
    ax.legend()
    plt.draw();
    plt.savefig(plots_folder+"icl_query_only.png", dpi=100, bbox_inches='tight')

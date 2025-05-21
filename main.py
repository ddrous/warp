#%%[markdown]

# ## Generative Recurrent Neural Networks in weight space

#%%

# %load_ext autoreload
# %autoreload 2

from utils import *
from loaders import *
from models import *

import jax
print("\n\nAvailable devices:", jax.devices())

from jax import config
config.update("jax_debug_nans", True)
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

model_type = config['model']['model_type']
if model_type == "wsm":
    print("\n\n+=+=+=+=+ Training Weight Space Model +=+=+=+=+\n")
elif model_type == "gru":
    print("\n\n+=+=+=+=+ Training Gated Recurrent Unit Model +=+=+=+=+\n")
elif model_type == "lstm":
    print("\n\n+=+=+=+=+ Training Long Short Term Memory Model +=+=+=+=+\n")
else:
    print("\n\n+=+=+=+=+ Training Unknown Model +=+=+=+=+\n")
    raise ValueError(f"Unknown model type: {model_type}")

seed = config['general']['seed']
main_key = jax.random.PRNGKey(seed)
np.random.seed(seed)
torch.manual_seed(seed)






#%%
train = config['general']['train']
classification = config['general']['classification']
data_path = config['general']['data_path']
save_path = config['general']['save_path']

### Create and setup the run and data folders
if train:
    if save_path is not None:
        run_folder = save_path
    else:
        # run_folder = make_run_folder('./runs/')
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

trainloader, validloader, testloader, data_props = make_dataloaders(data_folder, config)
nb_classes, seq_length, data_size, width = data_props

print("Total number training samples:", len(trainloader.dataset))

batch = next(iter(trainloader))
(in_sequence, times), output = batch
logger.info(f"Input sequence shape: {in_sequence.shape}")
logger.info(f"Labels/OutputSequence shape: {output.shape}")
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
repeat_datasets = ["lotka"]

res = (width, width, data_size)
dim0, dim1 = (0, 1)
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
            # axs[i, j].plot(in_sequence[idx, :, dim0], in_sequence[idx, :, dim1], color=colors[(i*j)%len(colors)])
            # axs[i, j].plot(output[idx, :, dim0], output[idx, :, dim1], color=colors[(i*j)%len(colors)], linestyle='--')
            ## Make 4 plots against time steps instead
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

    if not classification:
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
    
    else:           ## Classification task
        (X_true, times), Ys = batch

        Y_hat = model(X_true, times, key, inference_start=None)

        # print("\n\nPrint shapes of pred and true:", Y_hat.shape, Ys.shape, "\n\n")
        # Compute the cross entropy loss using Optax
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=Y_hat[:, -1], labels=Ys)

        # ## Manual cross entropy loss
        # Y_hat = jax.nn.softmax(Y_hat[:, -1], axis=-1)
        # Y_onehot = jax.nn.one_hot(Ys, num_classes=nb_classes)
        # loss = -jnp.sum(Y_onehot * jnp.log(Y_hat + 1e-10), axis=-1)

        loss = jnp.mean(loss)

        acc = jnp.mean(jnp.argmax(Y_hat[:, -1], axis=-1) == Ys)

        return loss, (acc,)


@eqx.filter_jit
def train_step(model, batch, opt_state, key):
    print('\nCompiling function "train_step" ...')

    (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, key)

    updates, opt_state = opt.update(grads, opt_state, model, value=loss)        ## For reduce on plateau loss accumulation
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, aux_data


@eqx.filter_jit
def forward_pass(model, X, times, key, inference_start=None):
    """ Jitted forward pass. """
    X_recons = model(X, times, key, inference_start)
    return X_recons


val_criterion = config['training']['val_criterion']
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
            if use_nll_loss:
                X_recons = X_recons[:, :, :data_size]
            if dataset in repeat_datasets:
                X_gt = X_labs_outs
            else:
                X_gt = X_true
            if val_criterion == "mse":
                loss_val = optax.l2_loss(X_recons, X_gt).mean()
            elif val_criterion == "mae":
                mloss_valse = optax.l1_loss(X_recons, X_gt).mean()
            elif val_criterion == "rmse":
                loss_val = optax.root_mean_squared_error(X_recons, X_gt).mean()
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
            else:
                raise ValueError(f"Unknown validation criterion for classification: {val_criterion}")

        cri_vals.append(loss_val)

    return np.mean(cri_vals), np.median(cri_vals), np.min(cri_vals)






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

    val_losses = []
    best_val_loss = np.inf
    best_val_loss_epoch = 0

    print_every = config['training']['print_every']
    save_every = config['training']['save_every']
    valid_every = config['training']['valid_every']

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
                logger.info(f"Average classification accuracy: {np.mean(aux_epoch)*100:.2f}%")

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
            val_mean_loss, val_median_loss, _ = eval_on_dataloader(model, validloader, inference_start=None, key=test_key)
            val_losses.append(val_mean_loss)

            logger.info(
                f"Epoch {epoch:-4d}/{nb_epochs:-4d}     Valid Loss   -Mean: {val_mean_loss:.6f},   -Median: {val_median_loss:.6f}"
            )

            ## Save the model with the lowest validation loss
            if epoch==0 or val_mean_loss<=np.min(val_losses[:-1]):
                eqx.tree_serialise_leaves(artefacts_folder+"model.eqx", model)
                # logger.info("Best model on validation set saved at epoch %d" % epoch)
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
        logger.info(f"Best model form epoch {best_val_loss_epoch} restored.")
    else:
        logger.info("No 'best' model found. Using the last model.")



else:
    model = eqx.tree_deserialise_leaves(artefacts_folder+"model.eqx", model)

    try:
        losses = np.load(artefacts_folder+"losses.npy")
        lr_scales = np.load(artefacts_folder+"lr_scales.npy")
        val_losses_raw = np.load(artefacts_folder+"val_losses.npy")
        val_losses = val_losses_raw['losses']
        best_val_loss_epoch = val_losses_raw['best_epoch'].item()
        best_val_loss = val_losses_raw['best_loss'].item()
    except:
        losses = []
        val_losses = []

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




## Plot the validation losses
val_ids = (np.arange(0, nb_epochs, valid_every).tolist() + [nb_epochs-1])[:len(val_losses)]
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax = sbplot(val_ids, val_losses, title=f"{val_criterion.upper()} on Valid Set at Various Epochs", x_label='Epoch', y_label=f'{val_criterion}', ax=ax, y_scale="log", linewidth=3);
plt.axvline(x=best_val_loss_epoch, color='r', linestyle='--', linewidth=3, label=f"Best {val_criterion.upper()}: {best_val_loss:.6f} at Epoch {best_val_loss_epoch}")
plt.legend(fontsize=16)
plt.draw();
plt.savefig(plots_folder+f"checkpoints_{val_criterion.lower()}.png", dpi=100, bbox_inches='tight')
logger.info(f"Best model found at epoch {best_val_loss_epoch} with {val_criterion}: {best_val_loss:.6f}")




# %% Other visualisation of the model

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
    min_val = -0.00
    max_val = 0.000003

    img = axs[0].imshow(untrained_model.As[0], cmap='viridis', vmin=min_val, vmax=max_val)
    axs[0].set_title("Untrained A")
    plt.colorbar(img, ax=axs[0], shrink=0.7)

    img = axs[1].imshow(model.As[0], cmap='viridis', vmin=min_val, vmax=max_val)
    axs[1].set_title("Trained A")
    plt.colorbar(img, ax=axs[1], shrink=0.7)
    plt.draw();
    plt.savefig(plots_folder+"A_matrices.png", dpi=100, bbox_inches='tight')

    ## Print the dynamic tanh_params attribute
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


# %% Visualising a few reconstruction samples

if not classification:
    ## Set inference mode to True
    visloader = NumpyLoader(testloader.dataset, batch_size=16, shuffle=True)
    nb_examples = len(visloader.dataset)    ## Actualy number of examples in the dataset

    nb_cols = 3 if use_nll_loss else 2
    fig, axs = plt.subplots(4, 4*nb_cols, figsize=(16*3, 16), sharex=True, constrained_layout=True)

    batch = next(iter(visloader))
    (xs_true, times), labels = batch

    inference_start = config['training']['inference_start']
    xs_recons = forward_pass(model=model, 
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
            x = xs_true[(i*4+j)%nb_examples]
            x_recons = xs_recons[(i*4+j)%nb_examples]
            x_full = labels[(i*4+j)%nb_examples]

            # if dataset in dynamics_datasets+repeat_datasets:
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
                axs[i, nb_cols*j].set_ylim([min(min_0, min_1)-eps, max(max_0, max_1)+eps])
                axs[i, nb_cols*j].plot(x_full[:, dim0], color=colors[(i*4+j)%len(colors)])
                axs[i, nb_cols*j].plot(x_full[:, dim1], color=colors[(i*4+j)%len(colors)], linestyle='-.')
            else:
                # axs[i, nb_cols*j].set_xlim([min_0-eps, max_0+eps])
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
            # elif dataset in dynamics_datasets and dataset not in repeat_datasets:
            elif dataset in repeat_datasets:
                axs[i, nb_cols*j+1].set_ylim([min(min_0, min_1)-eps, max(max_0, max_1)+eps])
                axs[i, nb_cols*j+1].plot(x_recons[:, dim0], color=colors[(i*4+j)%len(colors)])
                axs[i, nb_cols*j+1].plot(x_recons[:, dim1], color=colors[(i*4+j)%len(colors)], linestyle='-.')
            else:
                # axs[i, nb_cols*j+1].plot(x_recons, color=colors[int(labels[i*4+j])])
                axs[i, nb_cols*j+1].set_ylim([min(min_0, min_1)-eps, max(max_0, max_1)+eps])
                axs[i, nb_cols*j+1].plot(x_recons[:, dim0], color=colors[labels[(i*4+j)%nb_examples]%len(colors)])
                axs[i, nb_cols*j+1].plot(x_recons[:, dim1], color=colors[labels[(i*4+j)%nb_examples]%len(colors)], linestyle='-.')

            if i==0:
                axs[i, nb_cols*j+1].set_title("Recons", fontsize=40)
            # axs[i, nb_cols*j+1].axis('off')

            if use_nll_loss:
                # logger.info(f"Min/Max Uncertainty: {np.min(xs_uncert):.3f}, {np.max(xs_uncert):.3f}")
                if dataset in image_datasets:
                    to_plot = xs_uncert[i*4+j].reshape(res)
                    axs[i, nb_cols*j+2].imshow(to_plot, cmap='gray')
                # elif dataset in dynamics_datasets:
                    # axs[i, nb_cols*j+2].plot(to_plot[:, dim0], to_plot[:, dim1], color=colors[labels[i*4+j]%len(colors)])
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



#%% Get the MSE on the entire test set

if not classification:
    eval_loader = NumpyLoader(testloader.dataset, batch_size=len(testloader.dataset), shuffle=False)

    batch = next(iter(eval_loader))
    (xs_true, times), labels = batch


    print("Inferance starts at: ", inference_start)
    xs_recons = forward_pass(model=model, 
                        X=xs_true, 
                        times=times, 
                        key=test_key, 
                        inference_start=inference_start)

    # xs_true = xs_true[:, inference_start:, :]
    # xs_recons = xs_recons[:, inference_start:, :data_size]
    xs_recons = xs_recons[:, :, :data_size]

    def metrics(pred, true):

        ## Calculate the metrics after inference start
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
    logger.info(f"    - MSPE : {mspe:.6f}")



## %% Now dealing with classification tasks
## Unormalize the test data and the predictions before computing the metrics
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

    min_max = (trainloader.dataset.min_data, trainloader.dataset.max_data)
    xs_recons_unorm = unormalize_data(xs_recons, min_max)
    xs_true_unorm = unormalize_data(xs_true, min_max)

    ## Plot a few samples of the unormalized data
    plot_dim = np.random.randint(0, data_size, size=2)[0]
    plot_id = np.random.randint(0, len(xs_true_unorm), size=1)[0]
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
    logger.info(f"    - MSPE : {mspe:.6f}")



# # %% Let's plot the prediction and groundth, while graying our the context, for just one example
# if not classification:
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     time_steps = np.arange(xs_true_unorm.shape[1])
#     ax.plot(xs_true_unorm[plot_id, :, plot_dim], "r-", lw=2, label="Ground Truth")
#     ax.plot(xs_true_unorm[plot_id, :inference_start, plot_dim], color="yellow", lw=2)
#     ax.plot(time_steps[inference_start:], xs_recons_unorm[plot_id, inference_start:, plot_dim], lw=6, label="Prediction", color="orange")

#     ## PLot a vertical line for the inference start
#     ax.axvline(x=inference_start, color='k', linestyle='--', lw=3, label="Inference Start")

#     ax.legend(fontsize=14, loc="lower left")
    
#     ## Set xticks and yticks
#     ticks = [0, 32, 64, 96, 128, 160, 192]
#     ax.set_xticks(ticks)
#     ax.set_xticklabels(ticks)

#     ax.set_xlabel("Time Step $t$", fontsize=20)

#     plt.tight_layout()
#     plt.draw();
#     plt.savefig(plots_folder+"prediction.png", dpi=100, bbox_inches='tight')
#     plt.savefig(plots_folder+"prediction.pdf", dpi=100, bbox_inches='tight')













# %% Now dealing with classification tasks

if classification:

    ## Compute the accurary of the model on the test set in one go
    evalloader = NumpyLoader(testloader.dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    # evalloader = NumpyLoader(validloader.dataset, batch_size=16, shuffle=False)
    
    # batch = next(iter(evalloader))
    # (in_sequence, times), output = batch

    accs = []
    for i, batch in enumerate(evalloader):
        (in_sequence, times), output = batch
        Y_hat_raw = forward_pass(model, in_sequence, times, main_key)
        Y_hat = jnp.argmax(Y_hat_raw, axis=-1)
        # print("\nY_hat shape:", Y_hat.shape, "Y shape:", output.shape)
        acc = jnp.mean(Y_hat[:, -1] == output)

        accs.append(acc)

    accuracy = np.mean(accs)
    logger.info(f"Test set accuracy: {accuracy:.4f}")


    #### ## Visualise the model on the test set
    visloader = NumpyLoader(testloader.dataset, batch_size=16, shuffle=True)
    batch = next(iter(visloader))
    (in_sequence, times), output = batch

    ## PLot the first 16 seqences, with the title as the predicted class
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

                    # axs[i, j].plot(in_sequence[idx, :, dim0], color=colors[int(output[idx])%len(colors)], lw=3)
                    # axs[i, j].plot(in_sequence[idx, :, dim1], color=colors[int(output[idx])%len(colors)], linestyle='--', lw=3)

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

            # axs[i, j].set_ylim([-0.1, 1.1])
            axs[i, j].set_ylim([-0.1, nb_classes-1+0.1])

            ## Set the y ticks to be the class labels
            axs[i, j].set_yticks(np.arange(nb_classes))
            axs[i, j].set_yticklabels(np.arange(nb_classes))

            ## Set the x label as time step
            if i==3:
                axs[i, j].set_xlabel("Time Step", fontsize=22)

    plt.tight_layout()
    plt.suptitle(f"{dataset.upper()} Predicted Test Labels", fontsize=20, y=1.02)

    plt.draw();
    plt.savefig(plots_folder+"samples_test_labels.png", dpi=100, bbox_inches='tight')

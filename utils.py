import os
import numpy as np
import jax.tree as jtree
import jax.numpy as jnp
import equinox as eqx
import logging
import time
import sys
import matplotlib.pyplot as plt




############# Utility functions for JAX-implemented models ####################

def flatten_pytree(pytree):
    """ Flatten the leaves of a pytree into a single array. 
        Return the array, the shapes of the leaves and the tree_def. """

    leaves, tree_def = jtree.flatten(pytree)
    flat = jnp.concatenate([x.flatten() for x in leaves])
    shapes = [x.shape for x in leaves]
    return flat, shapes, tree_def

def unflatten_pytree(flat, shapes, tree_def):
    """ Reconstructs a pytree given its leaves flattened, their shapes, and the treedef. """

    leaves_prod = [0]+[np.prod(x) for x in shapes]

    lpcum = np.cumsum(leaves_prod)
    leaves = [flat[lpcum[i-1]:lpcum[i]].reshape(shapes[i-1]) for i in range(1, len(lpcum))]

    return jtree.unflatten(tree_def, leaves)

def count_params(module):
    """ Count the number of learnable parameters in an Equinox module. """
    return sum(x.size for x in jtree.leaves(eqx.filter(module, eqx.is_array)) if x is not None)



def compute_all_powers(A, n):
    m = A.shape[0]
    power_dict = {1: A.copy()}
    max_bit = n.bit_length()

    # Step 1: Precompute powers of 2
    current = A.copy()
    for i in range(1, max_bit):
        current = current @ current
        power_dict[2**i] = current

    # Step 2: Compose powers from binary expansion
    results = [jnp.eye(m)]
    for k in range(1, n):
        bits = [2**i for i in range(k.bit_length()) if (k >> i) & 1]
        prod = power_dict[bits[0]]
        for b in bits[1:]:
            prod = prod @ power_dict[b]
        results.append(prod)

    return results  # returns [A^0, A^1, ..., A^n-1]


def compute_kernel(A, B, T):
    d = A.shape[0]
    power_dict = {1: A}
    max_bit = T.bit_length()

    # Step 1: Precompute powers of 2
    current = A
    for i in range(1, max_bit):
        current = current @ current
        power_dict[2**i] = current

    # Step 2: Compose powers from binary expansion
    results = [jnp.eye(d)@B]
    for k in range(1, T):
        bits = [2**i for i in range(k.bit_length()) if (k >> i) & 1]
        prod = power_dict[bits[0]]
        for b in bits[1:]:
            prod = prod @ power_dict[b]
    
        results.append(prod @ B)

    return jnp.stack(results)  # returns [A^0@B, A^1@B, ..., A^n-1@B]





############# Other utility functions ####################

def seconds_to_hours(seconds):
    """ Convert seconds to hours, minutes, and seconds. """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds


def f1_score_macro(y_true, y_pred, nb_classes):
    """ Compute the macro F1 score. """
    f1s = []
    for cls in range(nb_classes):
        tp = jnp.sum((y_pred == cls) & (y_true == cls))
        fp = jnp.sum((y_pred == cls) & (y_true != cls))
        fn = jnp.sum((y_pred != cls) & (y_true == cls))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1s.append(f1)
    f1_macro = jnp.mean(jnp.array(f1s))
    return f1_macro


def make_run_folder(parent_path='./runs/'):
    """ Create a new folder for the run. """
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)

    run_folder = os.path.join(parent_path, time.strftime("%y%m%d-%H%M%S")+'/')
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        print("Created a new run folder at:", run_folder)

    return run_folder



def setup_logger(folder_path, training):
    """Set up a logger for training progress."""
    # Create logfile
    fname = "training.log" if training else "testing.log"
    log_filename = os.path.join(folder_path, fname)

    # Configure logger
    logger = logging.getLogger('training')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Started logging to {log_filename}")

    return logger


def setup_run_folder(folder_path, training=True):
    """ Copy the run scripts, and a logger in the run folder. """

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print("Created a new run folder at:", folder_path)

    # Save the run scripts in that folder
    os.system(f"cp main.py {folder_path}")
    os.system(f"cp utils.py {folder_path}")
    os.system(f"cp loaders.py {folder_path}")
    os.system(f"cp models.py {folder_path}")

    ## Create a folder for the chcekpoints results
    checkpoints_folder = folder_path+"checkpoints/"
    if not os.path.exists(checkpoints_folder):
        os.mkdir(checkpoints_folder)
        print(" Created a checkpoints folder at:", checkpoints_folder)

    ## Create a plot folder
    plots_folder = folder_path+"plots/"
    if not os.path.exists(plots_folder):
        os.mkdir(plots_folder)
        print(" Created a plots folder at:", plots_folder)

    ## Create a folder for the artefacts
    artefacts_folder = folder_path+"artefacts/"
    if not os.path.exists(artefacts_folder):
        os.mkdir(artefacts_folder)
        print(" Created a artefacts folder at:", artefacts_folder)

    logger = setup_logger(artefacts_folder, training)

    return logger, checkpoints_folder, plots_folder, artefacts_folder




## Wrapper function for matplotlib and seaborn
def sbplot(*args, 
           ax=None, 
           figsize=(6,3.5), 
           x_label=None, 
           y_label=None, 
           title=None, 
           x_scale='linear', 
           y_scale='linear', 
           xlim=None, 
           ylim=None, 
           **kwargs):
    if ax==None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    # sns.despine(ax=ax)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.plot(*args, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if "label" in kwargs.keys():
        ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    return ax

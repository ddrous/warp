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





############## Metrics ####################

# import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def _ensure_2d(a):
    """Return (n_samples, n_targets) shaped array."""
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _valid_mask_pair(y_true_col):
    """Mask where y_true is finite (not NaN/inf). y_pred NaNs will be set to zero."""
    return np.isfinite(y_true_col) 


def rank_correlation(y_true_col, y_pred_col):
    """Spearman rank correlation for a single target column with proper masking."""
    y_true_col = np.asarray(y_true_col)
    y_pred_col = np.asarray(y_pred_col)

    # Only mask based on y_true, set y_pred NaNs to zero
    mask = _valid_mask_pair(y_true_col)
    if mask.sum() < 2:
        return np.nan  # need at least 2 points for correlation

    y_pred_masked = y_pred_col[mask]
    y_pred_masked = np.where(np.isfinite(y_pred_masked), y_pred_masked, 0.0)
    
    corr = spearmanr(y_true_col[mask], y_pred_masked).correlation
    # spearmanr can still return nan if constant arrays
    return corr if np.isfinite(corr) else np.nan


def shape_ratio(y):
    """Calculate shape ratio (mean / std) ignoring NaNs; returns NaN if undefined."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]  # drop NaN/inf
    if y.size == 0:
        return np.nan
    std = np.std(y)
    if std == 0:
        return np.nan
    return np.mean(y) / std


def custom_metric(y_true, y_pred):
    """
    Per-target Spearman correlations (with masking) -> shape ratio of those correlations.
    """
    y_true = _ensure_2d(y_true)
    y_pred = _ensure_2d(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    correlations = []
    for i in range(y_true.shape[1]):
        corr = rank_correlation(y_true[:, i], y_pred[:, i])
        correlations.append(corr)

    # print(len(correlations), "correlations computed.")

    return shape_ratio(np.array(correlations, dtype=float))


def custom_metric_abs(y_true, y_pred):
    """
    Per-target Spearman correlations (with masking) -> shape ratio of those correlations.
    """
    y_true = _ensure_2d(y_true)
    y_pred = _ensure_2d(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    correlations = []
    for i in range(y_true.shape[1]):
        corr = rank_correlation(y_true[:, i], y_pred[:, i])
        correlations.append(corr)

    # print(len(correlations), "correlations computed.")

    return np.abs(np.mean(correlations, dtype=float))         ## Between 0 and 1

def _metric_per_target(y_true, y_pred, metric_fn, min_points=1):
    """
    Compute a sklearn metric per target after masking, then return the nan-mean across targets.
    metric_fn signature: (y_true_1d, y_pred_1d) -> float
    """
    y_true = _ensure_2d(y_true)
    y_pred = _ensure_2d(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    vals = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mask = _valid_mask_pair(yt)
        if mask.sum() >= min_points:
            try:
                # Set NaN predictions to zero for masked data
                yp_masked = yp[mask]
                yp_masked = np.where(np.isfinite(yp_masked), yp_masked, 0.0)
                vals.append(metric_fn(yt[mask], yp_masked))
            except Exception:
                vals.append(np.nan)
        else:
            vals.append(np.nan)
    # average across targets, ignoring NaNs
    return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan


def calculate_metrics(y_true, y_pred, prefix=""):
    """
    Calculate comprehensive evaluation metrics, masking NaNs/inf PER TARGET.
    Returns scalar metrics averaged across targets.
    """
    y_true = _ensure_2d(y_true)
    y_pred = _ensure_2d(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    metrics = {}

    # Standard regression metrics (per target, then average)
    metrics[f"{prefix}mse"] = _metric_per_target(y_true, y_pred, mean_squared_error, min_points=1)
    metrics[f"{prefix}mae"] = _metric_per_target(y_true, y_pred, mean_absolute_error, min_points=1)
    # r2 needs at least 2 points
    metrics[f"{prefix}r2"]  = _metric_per_target(y_true, y_pred, r2_score, min_points=2)

    # Custom metric (shape ratio of per-target Spearman correlations)
    metrics[f"{prefix}custom_metric"] = custom_metric(y_true, y_pred)

    # Sign accuracy per target, then average
    def _sign_acc(yt, yp):
        m = _valid_mask_pair(yt)
        if m.sum() == 0:
            return np.nan
        yp_masked = yp[m]
        yp_masked = np.where(np.isfinite(yp_masked), yp_masked, 0.0)
        return float(np.mean(np.sign(yt[m]) == np.sign(yp_masked)))

    metrics[f"{prefix}sign_accuracy"] = _metric_per_target(y_true, y_pred, _sign_acc, min_points=1)

    return metrics


def evaluate_model_by_lag(y_pred, y_true, model_name="Model"):
    """Evaluate model performance across lags, averaging metrics across targets within each lag."""

    ## Build y_true and y_pred dicts by splitting the last dimension into 4 lags
    y_true_vals = np.split(y_true, 4, axis=-1)
    y_pred_vals = np.split(y_pred, 4, axis=-1)
    y_true_dict = {f"lag_{i+1}": y_true_vals[i] for i in range(4)}
    y_pred_dict = {f"lag_{i+1}": y_pred_vals[i] for i in range(4)}

    results = {}
    for lag in [1, 2, 3, 4]:
        lag_key = f"lag_{lag}"
        if lag_key in y_true_dict and lag_key in y_pred_dict:
            y_true = y_true_dict[lag_key]
            y_pred = y_pred_dict[lag_key]
            lag_metrics = calculate_metrics(y_true, y_pred, f"lag{lag}_")
            results.update(lag_metrics)

            # Pretty print (guard NaNs)
            mse = lag_metrics.get(f"lag{lag}_mse", np.nan)
            mae = lag_metrics.get(f"lag{lag}_mae", np.nan)
            r2  = lag_metrics.get(f"lag{lag}_r2", np.nan)
            sa  = lag_metrics.get(f"lag{lag}_sign_accuracy", np.nan)
            cm  = lag_metrics.get(f"lag{lag}_custom_metric", np.nan)
            print(f"{model_name} - Lag {lag}: "
                f"MSE={mse:.6f} | MAE={mae:.6f} | R²={r2:.4f} | "
                f"Sign Acc={sa:.4f} | Custom Metric={cm:.4f}")
    return results


def mitsui_metric(y_pred, y_true):
    """ Returns only the custom metric defined above, across the batch.
    y_true and y_pred are (batch_size, time_steps, 4*106) arrays. """

    ## Focus on the final 89 steps
    y_true = y_true[:, -89:, :]
    y_pred = y_pred[:, -89:, :]

    sum_metric = 0.0
    for i in range(y_true.shape[0]):
        y_true_vals = np.split(y_true[i], 4, axis=-1)
        y_pred_vals = np.split(y_pred[i], 4, axis=-1)

        y_true_dict = {f"lag_{j+1}": y_true_vals[j] for j in range(4)}
        y_pred_dict = {f"lag_{j+1}": y_pred_vals[j] for j in range(4)}
        for lag in [1, 2, 3, 4]:
            lag_key = f"lag_{lag}"
            if lag_key in y_true_dict and lag_key in y_pred_dict:
                y_true_lag = y_true_dict[lag_key]
                y_pred_lag = y_pred_dict[lag_key]
                # lag_metric = custom_metric(y_true_lag, y_pred_lag)
                lag_metric = custom_metric_abs(y_true_lag, y_pred_lag)
                sum_metric += lag_metric if np.isfinite(lag_metric) else 0.0

    avg_metric = sum_metric / (y_true.shape[0] * 4)

    return avg_metric

    ## Jus retutn the correct metric between the two arrays
    # return np.correlate(y_true.flatten(), y_pred.flatten()) / (np.linalg.norm(y_true.flatten()) * np.linalg.norm(y_pred.flatten()))
    # return pearsonr(y_true.flatten(), y_pred.flatten())[0]


def log_return(data, lag=1):
    """Calculate log returns with specified lag; pad front with NaNs to keep length."""
    data = np.asarray(data, dtype=float)
    if data.size == 0 or lag <= 0:
        return np.full(data.shape, np.nan, dtype=float)
    if data.size <= lag:
        return np.full(data.shape, np.nan, dtype=float)
    log_ret = np.log(data[lag:] / data[:-lag])
    return np.concatenate([np.full(lag, np.nan, dtype=float), log_ret])







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

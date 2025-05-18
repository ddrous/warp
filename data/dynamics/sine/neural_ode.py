
#%%[markdown]
# # Simple Neural ODE framework for the Mass-Spring-Damper system


#%%
%load_ext autoreload
%autoreload 2

import jax

print("\n############# SineWave with Neural ODE #############\n")
print("Available devices:", jax.devices())

from jax import config
##  Debug nans
# config.update("jax_debug_nans", True)

import jax.numpy as jnp

import numpy as np
from scipy.integrate import solve_ivp

import equinox as eqx
import diffrax

import matplotlib.pyplot as plt

# from neuralhub.utils import *
# from neuralhub.integrators import *

import optax
from functools import partial
import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="notebook", style="ticks", palette="deep", font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})


#%%

SEED = 2026
# SEED = time.time_ns() % 2**15
print(f"Seed: {SEED}")

##
def seconds_to_hours(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds

## Simply returns a suitable key for all jax operations
def get_new_key(key=None, num=1):
    if key is None:
        print("WARNING: No key provided, using time as seed")
        key = jax.random.PRNGKey(time.time_ns())

    elif isinstance(key, int):
        key = jax.random.PRNGKey(key)

    keys = jax.random.split(key, num=num)

    return keys if num > 1 else keys[0]

## Wrapper function for matplotlib and seaborn
def sbplot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, **kwargs):
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



## Integrator hps
# integrator = rk4_integrator
# integrator = dopri_integrator
integrator = diffrax.Tsit5()
# integrator = diffrax.Euler()

## Optimiser hps
init_lr = 1e-3
decay_rate = 0.9

## Training hps
print_every = 100
nb_epochs = 1000
# batch_size = 128*10
skip_steps = 1

#%%

problem_type = "huge" # "tiny", ""small "medium", "large", "huge"

data = np.load(f"{problem_type}/train.npy")[:, ::skip_steps, :]
print("Train data shape:", data.shape)

nb_samples, nb_steps, dim = data.shape

t_eval = jnp.linspace(0, 1, nb_steps)
batch_size = nb_samples

# %%

class Processor(eqx.Module):
    # layers: list
    # physics: jnp.ndarray
    network: jnp.ndarray

    def __init__(self, in_size, out_size, key=None):
        # keys = get_new_key(key, num=3)
        # self.layers = [eqx.nn.Linear(in_size, 8, key=keys[0]), jax.nn.softplus,
        #                 eqx.nn.Linear(8, 8, key=keys[1]), jax.nn.softplus,
        #                 eqx.nn.Linear(8, out_size, key=keys[2]) ]

        # self.matrix = jax.random.uniform(keys[0], (in_size, out_size), minval=-1, maxval=0)
        # self.matrix = jnp.array([[0., 0.], [0., 0.]])
        self.network = eqx.nn.MLP(
            in_size=in_size+1,
            out_size=out_size,
            width_size=48,
            depth=3,
            activation=jax.nn.swish,
            key=key,)

    def __call__(self, t, x, args):

        # ## Neural Net contribution
        # y = x
        # for layer in self.layers:
        #     y = layer(y)
        # return y

        # return self.matrix @ x

        # return self.network(x)
        # return self.network(x) * jnp.cos(2 * jnp.pi * t)

        t = jnp.expand_dims(t, axis=-1)
        return self.network(jnp.concatenate([t, x], axis=-1))


class NeuralODE(eqx.Module):
    data_size: int
    vector_field: eqx.Module

    def __init__(self, data_size, key=None):
        self.data_size = data_size
        self.vector_field = Processor(data_size, data_size, key=key)

    def __call__(self, x0s, t_eval):

        def integrate(y0):
            sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(self.vector_field),
                    # diffrax.Tsit5(),
                    integrator,
                    t0=t_eval[0],
                    t1=t_eval[-1],
                    dt0=t_eval[1]-t_eval[0],
                    y0=y0,
                    # stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                    saveat=diffrax.SaveAt(ts=t_eval),
                    # adjoint=diffrax.RecursiveCheckpointAdjoint(),
                    max_steps=4096
                )
            return sol.ys, sol.stats["num_steps"]

            # sol = RK4(self.vector_field, 
            #           (t_eval[0], t_eval[-1]), 
            #           y0, 
            #           (coeffs.lambdas, coeffs.gammas), 
            #           t_eval=t_eval, 
            #           subdivisions=4)
            # return sol, len(t_eval)*4

        trajs, nb_fes = eqx.filter_vmap(integrate)(x0s)
        # trajs, nb_fes = integrate(x0s)
        return trajs, jnp.sum(nb_fes)



# %%

model_keys = get_new_key(SEED, num=2)
model = NeuralODE(data_size=1, key=model_keys[0])







# %%


# def params_norm(params):
#     return jnp.array([jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(params)]).sum()

# def l2_norm(X, X_hat):
#     total_loss = jnp.mean((X - X_hat)**2, axis=-1)   ## Norm of d-dimensional vectors
#     return jnp.sum(total_loss) / (X.shape[-2])

# %%

### ==== Vanilla Gradient Descent optimisation ==== ####

def loss_fn(model, X):
    print('\nCompiling function "loss_fn" ...\n')

    X_hat, _ = model(X[:, 0, :], t_eval)

    # return jnp.mean((X[...,-1] - X_hat[...,-1])**2)
    return jnp.mean((X - X_hat)**2)


@eqx.filter_jit
def train_step(model, batch, opt_state):
    print('\nCompiling function "train_step" ...\n')

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)

    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


total_steps = nb_epochs

# sched = optax.exponential_decay(init_lr, total_steps, decay_rate)
# sched = optax.linear_schedule(init_lr, 0, total_steps, 0.25)
# sched = optax.piecewise_constant_schedule(init_value=init_lr,
#                                             boundaries_and_scales={int(total_steps*0.25):0.5, 
#                                                                     int(total_steps*0.5):0.2,
#                                                                     int(total_steps*0.75):0.5})
sched = init_lr

start_time = time.time()


print(f"\n\n=== Beginning Training ... ===")

opt = optax.adabelief(sched)
# opt = optax.sgd(sched)

# params, static  = eqx.partition(model, eqx.is_array)
# opt_state = opt.init(params)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

batch_size = 16
losses = []
for epoch in range(nb_epochs):

    nb_batches = 0
    loss_sum = 0.
    for i in range(0, data.shape[1], batch_size):
        batch = data[i:i+batch_size,...]

        model, opt_state, loss = train_step(model, batch, opt_state)

        loss_sum += loss
        nb_batches += 1

    loss_epoch = loss_sum/nb_batches
    losses.append(loss_epoch)

    if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
        print(f"    Epoch: {epoch:-5d}      Loss: {loss_epoch:.8f}", flush=True)


wall_time = time.time() - start_time
time_in_hmsecs = seconds_to_hours(wall_time)
print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)


fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
# ax = sbplot(losses, x_label='Epoch', y_label='L2', y_scale="log", title=f'Loss for environment {e}', ax=ax);
ax = sbplot(losses, x_label='Epoch', y_label='L2', y_scale="log", title='Loss', ax=ax);
# plt.savefig(f"{problem_type}/loss_{SEED:05d}.png", dpi=300, bbox_inches='tight')
# plt.show()
plt.legend()
fig.canvas.draw()
fig.canvas.flush_events()


## Save the losses
# np.save(f"{problem_type}/losses_{SEED:05d}.npy", np.array(losses))


# %%
def test_model(model, batch):
    X0 = batch
    X_hat, _ = model(X0, t_eval)
    return X_hat

i = 0

X = np.load(f"{problem_type}/test.npy")
t = t_eval

X_hat = test_model(model, X[:, 0, :])
# print("Test data shape:", X_hat.shape)

# X= X[i, :, :]
# X_hat = X_hat[i, :, :]

ax = sbplot(t, X_hat[i, :,:], x_label='Time', label='Prediction', title=f'Trajectories, {i}')
ax = sbplot(t, X[i, :,:], "+", color="grey", label='Ground Truth', x_label='Time', title=f'Trajectories, {i}', ax=ax)

# plt.savefig(f"data/coda_test_env{e}_traj{i}.png", dpi=300, bbox_inches='tight')
# plt.savefig(f"{problem_type}/predicted_trajs_{SEED:05d}.png", dpi=300, bbox_inches='tight')


#%% 

## Calculate the MSE and MAE on the test set
print("Metrics on the test set ...")
mse = jnp.mean((X - X_hat)**2)
mae = jnp.mean(jnp.abs(X - X_hat))
mape = jnp.mean(jnp.abs(X - X_hat) / (jnp.abs(X) + 0.))
print(f"    - MSE: {mse:.8f}")
print(f"    - MAE: {mae:.8f}")
print(f"    - MAPE: {mape:.8f}")


# %% [markdown]

# # Conclusion
# 
# 





















import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, LogLocator

import matplotlib as mpl
sb = sns
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



# Data from the table
nb_samples = np.array([1, 10, 100, 1000, 10000])
wsm_mse = np.array([0.030636, 0.000277, 0.000951, 0.000474, 0.000615])
wsm_mape = np.array([2.186242, 0.123913, 0.278747, 0.227525, 0.280586])*1
node_mse = np.array([0.81572676, 0.03518130, 0.03197455, 0.03197455, 0.03197455])
node_mape = np.array([5.19721508, 1.27983057, 1.29165757, 1.29165757, 1.29165757])*1

# Create a figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

# Define colors with a professional color palette
colors = sns.color_palette("colorblind", 4)

# Plot MSE on the first axis (left subplot)
ax1.plot(np.log10(nb_samples), wsm_mse, 'o-', color=colors[0], linewidth=2.5, 
         markersize=8, label='WARP', alpha=0.9)
ax1.plot(np.log10(nb_samples), node_mse, 's-', color=colors[1], linewidth=2.5, 
         markersize=8, label='Neural ODE', alpha=0.9)

# Plot MAPE on the second axis (right subplot)
ax2.plot(np.log10(nb_samples), wsm_mape, 'o-', color=colors[0], linewidth=2.5, 
         markersize=8, label='WARP', alpha=0.9)
ax2.plot(np.log10(nb_samples), node_mape, 's-', color=colors[1], linewidth=2.5, 
         markersize=8, label='Neural ODE', alpha=0.9)

# Set log scale for the MSE axis
ax1.set_yscale('log')

# Set y-axis labels
ax1.set_ylabel('MSE', fontsize=24, fontweight='bold')
ax2.set_ylabel('MAPE(%)', fontsize=24, fontweight='bold')

# Set shared x-axis label
# fig.text(0.5, 0.02, '# Samples', fontsize=14, fontweight='bold', ha='center')
ax1.set_xlabel('# Samples', fontsize=20, fontweight='bold')
ax2.set_xlabel('# Samples', fontsize=20, fontweight='bold')

## Set horizon legend on ax1
# ax1.legend(loc='upper center', bbox_to_anchor=(0.49, 1.15), ncol=2, fontsize=12, frameon=True, shadow=True)

# ax1.legend(loc='upper center', fontsize=14, frameon=True, shadow=True)
ax2.legend(loc='upper center', fontsize=15, frameon=True, shadow=True)

# Set the title
# fig.suptitle('Model Performance vs. Number of Training Samples', fontsize=16, fontweight='bold', y=0.98)

# Set custom x-ticks for log scale of sample sizes
for ax in [ax1, ax2]:
    # ax.set_xticks(np.log10(nb_samples))
    # ax.set_xticklabels([f'{int(x)}' for x in nb_samples])
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.set_xticks(np.log10(nb_samples))
    ax.set_xticklabels(["1", "10", "100", "1k", "10k"])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    

    # Add a box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

# Format the tick labels to avoid scientific notation
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_major_formatter(ScalarFormatter())

# Move y-axis ticks to the right side for MAPE plot
ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()


# Adjust layout and save
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
# fig.savefig('data_efficiency.png', dpi=100, bbox_inches='tight')
fig.savefig('data_efficiency.pdf', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

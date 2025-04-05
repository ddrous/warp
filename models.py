#%%
from utils import flatten_pytree, unflatten_pytree, count_params

import jax
import jax.numpy as jnp
import equinox as eqx





class RootMLP(eqx.Module):
    """ Root network f: t -> x_t, whose weights are the latent space of the WSM """
    network: eqx.Module
    props: any              ## Properties of the root network
    predict_uncertainty: bool
    apply_tanh_uncertainty: bool

    def __init__(self, 
                 data_size, 
                 width_size, 
                 depth, 
                 activation=jax.nn.relu,
                 input_prev_data=False,              ## Whether to inlude as input the previous data point
                 predict_uncertainty=True,           ## Predict the std in addition to the mean
                 apply_tanh_uncertainty=True,        ## Apply the tanh activation to the std, before the softplus
                 key=None):

        input_dim = 1+data_size if input_prev_data else 1
        output_dim = 2*data_size if predict_uncertainty else data_size

        self.network = eqx.nn.MLP(input_dim, output_dim, width_size, depth, activation, key=key)
        self.props = (input_dim, output_dim, width_size, depth, activation)
        self.predict_uncertainty = predict_uncertainty
        self.apply_tanh_uncertainty = apply_tanh_uncertainty

    def __call__(self, tx):
        out = self.network(tx)

        if not self.predict_uncertainty:
            return jnp.tanh(out)
        else:
            mean, std = jnp.split(out, 2, axis=-1)
            mean = jnp.tanh(mean)
            if self.apply_tanh_uncertainty:
                std = jnp.tanh(std)
            return jnp.concatenate([mean, jax.nn.softplus(std)], axis=-1)


class WSM_RNN(eqx.Module):
    """ Weight Space Seq2Seq Model, with RNN transition function """
    As: jnp.ndarray
    Bs: jnp.ndarray
    thetas: jnp.ndarray
    root_utils: list

    data_size: int
    input_prev_data: bool
    time_as_channel: bool
    forcing_prob: float
    weights_lim: float
    noise_theta_init: float

    def __init__(self, 
                 data_size, 
                 width_size, 
                 depth, 
                 activation="relu",
                 input_prev_data=False,
                 predict_uncertainty=True,
                 apply_tanh_uncertainty=True,
                 time_as_channel=True,
                 forcing_prob=1.0,
                 weights_lim=None,
                 noise_theta_init=None,             ## Noise to be added to the initial theta 
                 nb_wsm_layers=1,                  ## TODO, to be implemented as in Fig 2. of https://arxiv.org/abs/2202.07022
                 key=None):

        keys = jax.random.split(key, num=nb_wsm_layers)
        builtin_fns = {"relu":jax.nn.relu, "tanh":jax.nn.tanh, 'softplus':jax.nn.softplus, 'swish':jax.nn.swish}
        thetas = []
        root_utils = []
        As = []
        Bs = []
        for i in range(nb_wsm_layers):
            root = RootMLP(data_size, 
                           width_size, 
                           depth, 
                           builtin_fns[activation], 
                           input_prev_data=input_prev_data, 
                           predict_uncertainty=predict_uncertainty,
                           apply_tanh_uncertainty=apply_tanh_uncertainty,
                           key=keys[i])
            params, static = eqx.partition(root, eqx.is_array)
            weights, shapes, treedef = flatten_pytree(params)
            root_utils.append((shapes, treedef, static, root.props))

            thetas.append(weights)                              ## The latent space of the model

            latent_size = weights.shape[0]
            As.append(jnp.eye(latent_size))                     ## The most stable matrix: identity

            B_out_dim = data_size+1 if time_as_channel else data_size
            B = jnp.zeros((latent_size, B_out_dim))
            B += jax.random.normal(keys[i], B.shape)*1e-2       ## Initial perturbation to avoid getting stuck TODO
            Bs.append(B)

        self.root_utils = root_utils
        self.thetas = thetas
        self.As = As
        self.Bs = Bs

        self.data_size = data_size
        self.input_prev_data = input_prev_data
        self.time_as_channel = time_as_channel
        self.forcing_prob = forcing_prob
        self.weights_lim = weights_lim
        self.noise_theta_init = noise_theta_init

    def __call__(self, xs, ts, k, inference_start=None):
        """ Forward pass of the model on batch of sequences
            xs: (batch, time, data_size)
            ts: (batch, time)
            k:  (key_dim)
            inference_start: whether/when to use the model in autoregressive mode
            """

        def forward(xs_, ts_, k_):
            """ Forward pass on a single sequence """

            def f(carry, input_signal):
                thet, x_hat, x_prev, t_prev = carry
                x_true, t_curr, key = input_signal
                delta_t = t_curr - t_prev

                A = self.As[0]
                B = self.Bs[0]
                root_utils = self.root_utils[0]

                if inference_start is not None:
                    x_t = jnp.where(t_curr<=inference_start/ts_.shape[0], x_true, x_hat)
                else:
                    x_t = jnp.where(jax.random.bernoulli(key, self.forcing_prob), x_true, x_hat)

                if self.time_as_channel:
                    x_t = jnp.concatenate([t_curr, x_t], axis=-1)
                    x_prev = jnp.concatenate([t_prev, x_prev], axis=-1)

                thet_next = A@thet + B@(x_t - x_prev)     ## Key step

                if self.weights_lim is not None:
                    thet_next = jnp.clip(thet_next, -self.weights_lim, self.weights_lim)

                shapes, treedef, static, _ = root_utils
                params = unflatten_pytree(thet_next, shapes, treedef)
                root_fun = eqx.combine(params, static)
                root_in = t_curr+delta_t
                if self.input_prev_data:
                    root_in = jnp.concatenate([root_in, x_prev], axis=-1)
                x_next = root_fun(root_in)                                  ## Evaluated at the next time step

                x_next_mean = x_next[:x_true.shape[0]]

                return (thet_next, x_next_mean, x_hat, t_curr), (x_next, )

            ## Call the scan function
            theta_init = self.thetas[0]
            if self.noise_theta_init is not None:
                theta_init += jax.random.normal(k_, theta_init.shape)*self.noise_theta_init

            keys = jax.random.split(k_, xs_.shape[0])

            _, (xs_hat, ) = jax.lax.scan(f, (theta_init, xs_[0], xs_[0], -ts_[1:2]), (xs_, ts_[:, None], keys))

            return xs_hat

        ## Batched version of the forward pass
        ks = jax.random.split(k, xs.shape[0])
        return eqx.filter_vmap(forward)(xs, ts, ks)




def make_model(key, data_size, config):
    """ Make a model using the given key and kwargs """

    model_type = config['model']['model_type']

    if model_type == "wsm-rnn":
        model_args = {
            "data_size": data_size,
            "width_size": config['model']['mlp_width_size'],
            "depth": config['model']['mlp_depth'],
            "activation": config['model']['activation'],
            "input_prev_data": config['model']['input_prev_data'],
            "predict_uncertainty": config['training']['use_nll_loss'],
            "apply_tanh_uncertainty": config['model']['apply_tanh_uncertainty'],
            "time_as_channel": config['model']['time_as_channel'],
            "forcing_prob": config['model']['forcing_prob'],
            "weights_lim": config['model']['weights_lim'],
            "noise_theta_init": config['model']['noise_theta_init']
        }

        model = WSM_RNN(key=key, **model_args)
        print(f"Number of learnable parameters in the root network: {count_params((model.thetas,))/1000:3.1f} k")
        print(f"Number of learnable parameters for the seqtoseq's transition: {count_params((model.As, model.Bs))/1000:3.1f} k")
    elif model_type == "wsm-lstm":
        raise NotImplementedError("LSTM transition model not implemented yet")
    elif model_type == "wsm-gru":
        raise NotImplementedError("GRU transition model not implemented yet")
    elif model_type == "rnn":
        raise NotImplementedError("Standard RNN not implemented yet")
    elif model_type == "lstm":
        raise NotImplementedError("Standard LSTM not implemented yet")
    elif model_type == "gru":
        raise NotImplementedError("Standard GRU not implemented yet")
    elif model_type == "s4":
        raise NotImplementedError("S4 not implemented yet")     ## from https://github.com/ddrous/annotated-s4
    elif model_type == "mamba":
        raise NotImplementedError("Transformer not implemented yet")        ## https://github.com/ddrous/mamba-jax
    elif model_type == "transformer":
        raise NotImplementedError("Transformer not implemented yet")

    print(f"Number of learnable parameters in the model: {count_params(model)/1000:3.1f} k")

    return model







if __name__ == "__main__":
    
    ## Test a dummy model, with some dummy inputs
    key = jax.random.PRNGKey(0)
    model_args = {
        "data_size": 3,
        "width_size": 64,
        "depth": 2,
        "activation": "relu",
        "input_prev_data": False,
        "predict_uncertainty": True,
        "apply_tanh_uncertainty": True,
        "time_as_channel": True,
        "forcing_prob": 1.0,
        "weights_lim": 0.1,
        "noise_theta_init": None
    }

    model = make_model(key, "wsm-rnn", model_args)

    xs = jax.random.normal(key, (10, 200, 3))
    ts = jax.random.uniform(key, (10, 200))
    k = jax.random.PRNGKey(0)
    out = model(xs, ts, k, inference_start=0.3)
    print(out.shape)


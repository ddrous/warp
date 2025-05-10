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
    final_activation: any

    def __init__(self, 
                 data_size, 
                 width_size, 
                 depth, 
                 activation=jax.nn.relu,
                 final_activation=jax.nn.tanh,
                 input_prev_data=False,              ## Whether to inlude as input the previous data point
                 predict_uncertainty=True,           ## Predict the std in addition to the mean
                 key=None):

        input_dim = 1+data_size if input_prev_data else 1
        output_dim = 2*data_size if predict_uncertainty else data_size

        self.network = eqx.nn.MLP(input_dim, output_dim, width_size, depth, activation, key=key)
        self.props = (input_dim, output_dim, width_size, depth, activation)
        self.predict_uncertainty = predict_uncertainty
        self.final_activation = final_activation

    def __call__(self, tx, std_lb=None, dtanh=None):
        out = self.network(tx)

        if not self.predict_uncertainty:
            if self.final_activation is not None:
                out = self.final_activation(out)
            elif dtanh is not None:
                a, b, alpha, beta = dtanh
                out = alpha*jnp.tanh((out - b) / a) + beta
            else:
                pass

            return out

        else:
            mean, std = jnp.split(out, 2, axis=-1)

            if self.final_activation is not None:
                mean = self.final_activation(mean)
            elif dtanh is not None:
                a, b, alpha, beta = dtanh
                mean = alpha*jnp.tanh((mean - b) / a) + beta
            else:
                pass

            std = jax.nn.softplus(std)
            if std_lb is not None:
                std = jnp.clip(std, std_lb, None)

            return jnp.concatenate([mean, std], axis=-1)



class RootMLP_Classif(eqx.Module):
    """ Root network f: t -> y_t, whose weights are the latent space of the WSM """
    network: eqx.Module
    props: any              ## Properties of the root network

    def __init__(self, 
                 nb_classes, 
                 width_size, 
                 depth, 
                 activation=jax.nn.relu,
                 key=None):

        input_dim = 1
        output_dim = nb_classes

        self.network = eqx.nn.MLP(input_dim, output_dim, width_size, depth, activation, key=key)
        self.props = (input_dim, output_dim, width_size, depth, activation)

    def __call__(self, tx):
        out = self.network(tx)
        return out



class GradualMLP(eqx.Module):
    layers: list

    def __init__(self, input_dim, output_dim, hidden_layers=2, activation=jax.nn.tanh, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)

        if hidden_layers == 2:
            ## We want two intermediate layers: with input neurons gradually decreasing to output_dim
            hidden_size1 = int(2/3*input_dim + 1/3*output_dim)
            hidden_size2 = int(1/3*input_dim + 2/3*output_dim)
            in_layer = eqx.nn.Linear(input_dim, hidden_size1, key=keys[0])
            hidden_layer = eqx.nn.Linear(hidden_size1, hidden_size2, key=keys[1])
            out_layer = eqx.nn.Linear(hidden_size2, output_dim, key=keys[2])
            self.layers = [in_layer, activation, hidden_layer, activation, out_layer]

        elif hidden_layers == 1:
            ## We want one intermediate layer: with input neurons gradually decreasing to output_dim
            hidden_size = int(0.5*input_dim + 0.5*output_dim)
            in_layer = eqx.nn.Linear(input_dim, hidden_size, key=keys[0])
            out_layer = eqx.nn.Linear(hidden_size, output_dim, key=keys[1])
            self.layers = [in_layer, activation, out_layer]

        elif hidden_layers == 0:
            ## No intermediate layer
            in_layer = eqx.nn.Linear(input_dim, output_dim, key=keys[0])
            self.layers = [in_layer]

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y




class WSM(eqx.Module):
    """ Weight Space Seq2Seq Model, with RNN transition function """
    As: jnp.ndarray
    Bs: jnp.ndarray
    thetas_init: eqx.Module
    root_utils: list

    data_size: int
    input_prev_data: bool
    time_as_channel: bool
    forcing_prob: float
    weights_lim: float
    noise_theta_init: float
    init_state_layers: int

    classification: bool
    ar_train: bool

    std_lower_bound: float
    dtanh_params: jnp.ndarray

    def __init__(self, 
                 data_size, 
                 width_size, 
                 depth, 
                 activation="relu",
                 final_activation="tanh",
                 nb_classes=None,
                 init_state_layers=2,
                 input_prev_data=False,
                 predict_uncertainty=True,
                 time_as_channel=True,
                 forcing_prob=1.0,
                 std_lower_bound=None,
                 weights_lim=None,
                 noise_theta_init=None,             ## Noise to be added to the initial theta 
                 nb_wsm_layers=1,                  ## TODO, to be implemented as in Fig 2. of https://arxiv.org/abs/2202.07022
                 autoregressive_train=True,
                 key=None):

        keys = jax.random.split(key, num=nb_wsm_layers)
        builtin_fns = {"relu":jax.nn.relu, "tanh":jax.nn.tanh, 'softplus':jax.nn.softplus, 'swish':jax.nn.swish, "identity": lambda x: x}
        thetas_init = []
        root_utils = []
        As = []
        Bs = []
        for i in range(nb_wsm_layers):
            if nb_classes is None:          ## Regresion problem
                root = RootMLP(data_size, 
                            width_size, 
                            depth, 
                            builtin_fns[activation], 
                            final_activation=builtin_fns[final_activation] if isinstance(final_activation, str) else None,
                            input_prev_data=input_prev_data, 
                            predict_uncertainty=predict_uncertainty,
                            key=keys[i])
            else:                           ## Classification problem
                root = RootMLP_Classif(nb_classes, 
                            width_size, 
                            depth, 
                            builtin_fns[activation], 
                            key=keys[i])

            params, static = eqx.partition(root, eqx.is_array)
            weights, shapes, treedef = flatten_pytree(params)
            root_utils.append((shapes, treedef, static, root.props))

            if init_state_layers is None:
                thetas_init.append(weights)                              ## The latent space of the model
            else:
                thetas_init.append(GradualMLP(data_size, weights.shape[0], init_state_layers, builtin_fns[activation], key=keys[0]))

            latent_size = weights.shape[0]
            As.append(jnp.eye(latent_size))                     ## The most stable matrix: identity

            B_out_dim = data_size+1 if time_as_channel else data_size
            B = jnp.zeros((latent_size, B_out_dim))
            # B += jax.random.normal(keys[i], B.shape)*1e-3         ## Initial perturbation to avoid getting stuck TODO
            Bs.append(B)

        self.root_utils = root_utils
        self.thetas_init = thetas_init
        self.As = As
        self.Bs = Bs

        self.data_size = data_size
        self.input_prev_data = input_prev_data
        self.time_as_channel = time_as_channel
        self.forcing_prob = forcing_prob
        self.weights_lim = weights_lim
        self.noise_theta_init = noise_theta_init
        self.init_state_layers = init_state_layers

        self.classification = nb_classes is not None
        self.ar_train = autoregressive_train

        if self.classification and self.ar_train:
            raise ValueError("The WSM model is not compatible with autoregressive training for classification tasks.")

        self.std_lower_bound = std_lower_bound

        if isinstance(final_activation, list):      ## and len(final_activation) == 4
            self.dtanh_params = jnp.array(final_activation, dtype=jnp.float32)
        else:
            self.dtanh_params = None



    def __call__(self, xs, ts, k, inference_start=None):
        """ Forward pass of the model on batch of sequences
            xs: (batch, time, data_size)
            ts: (batch, time)
            k:  (key_dim)
            inference_start: whether/when to use the model in autoregressive mode
            """

        if self.classification:
            return self.non_ar_call(xs, ts, k)
        else:       ## Regression task
            if (self.ar_train) or (inference_start is not None):
                return self.ar_call(xs, ts, k, inference_start=inference_start)
            else:
                return self.non_ar_call(xs, ts, k)


    def ar_call(self, xs, ts, k, inference_start=None):
        """ Forward pass of the model on batch of sequences, ==autoregressively==
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
                x_next = root_fun(root_in, self.std_lower_bound, self.dtanh_params)                                  ## Evaluated at the next time step

                x_next_mean = x_next[:x_true.shape[0]]

                return (thet_next, x_next_mean, x_hat, t_curr), (x_next, )

            if self.init_state_layers is None:
                theta_init = self.thetas_init[0]
            else:
                theta_init = self.thetas_init[0](xs_[0])

            if self.noise_theta_init is not None:
                theta_init += jax.random.normal(k_, theta_init.shape)*self.noise_theta_init

            keys = jax.random.split(k_, xs_.shape[0])

            _, (xs_hat, ) = jax.lax.scan(f, (theta_init, xs_[0], xs_[0], ts_[0:1]), (xs_, ts_[:, None], keys))

            return xs_hat

        ## Batched version of the forward pass
        ks = jax.random.split(k, xs.shape[0])
        return eqx.filter_vmap(forward)(xs, ts, ks)


    def non_ar_call(self, xs, ts, k):
        """ Forward pass of the model on batch of sequences, ==non-autoregressively==
            xs: (batch, time, data_size)
            ts: (batch, time)
            k:  (key_dim)
            """

        def forward(xs_, ts_, k_):
            """ Forward pass on a single sequence """

            def f(carry, input_signal):
                thet, x_prev, t_prev = carry
                x_true, t_curr = input_signal

                A = self.As[0]
                B = self.Bs[0]

                x_t = x_true
                if self.time_as_channel:
                    x_t = jnp.concatenate([t_curr, x_t], axis=-1)
                    x_prev = jnp.concatenate([t_prev, x_prev], axis=-1)

                thet_next = A@thet + B@(x_t - x_prev)     ## Key step

                if self.weights_lim is not None:
                    thet_next = jnp.clip(thet_next, -self.weights_lim, self.weights_lim)

                return (thet_next, x_true, t_curr), (thet_next, )

            ## Call the scan function
            if self.init_state_layers is None:
                theta_init = self.thetas_init[0]
            else:
                theta_init = self.thetas_init[0](xs_[0])

            if self.noise_theta_init is not None:
                theta_init += jax.random.normal(k_, theta_init.shape)*self.noise_theta_init

            _, (theta_outs, ) = jax.lax.scan(f, (theta_init, xs_[0], ts_[0:1]), (xs_, ts_[:, None]))

            @eqx.filter_vmap
            def apply_theta(theta, t_curr, x_curr):
                delta_t = ts_[1] - ts_[0]
                root_utils = self.root_utils[0]
                shapes, treedef, static, _ = root_utils
                params = unflatten_pytree(theta, shapes, treedef)
                root_fun = eqx.combine(params, static)
                root_in = t_curr+delta_t

                if self.input_prev_data:
                    root_in = jnp.concatenate([root_in, x_curr], axis=-1)

                if not self.classification:
                    x_next = root_fun(root_in, self.std_lower_bound, self.dtanh_params)                                                 ## Evaluated at the next time step
                else:
                    x_next = root_fun(root_in)                                      ## Evaluated at the next time step
                return x_next

            xs_hat = apply_theta(theta_outs, ts_[:, None], xs_)

            return xs_hat

        ## Batched version of the forward pass
        ks = jax.random.split(k, xs.shape[0])
        return eqx.filter_vmap(forward)(xs, ts, ks)


    def tbptt_non_ar_call(self, xs, ts, k):
        """ Forward pass of the model on batch of sequences with truncated backpropagation through time
        
        Args:
            xs: (batch, time, data_size) - Input sequences
            ts: (batch, time) - Time steps
            k: (key_dim) - Random key
            num_chunks: int - Number of chunks to split sequences into
        
        Returns:
            xs_hat: (batch, time, data_size) - Predicted sequences
        """
        num_chunks=4        ## TODO: make this a parameter of the model

        def forward_tbptt(xs_, ts_, k_):
            """ Forward pass on a single sequence with truncated backpropagation """
            seq_len = xs_.shape[0]
            
            # Calculate chunk size - ensure it's at least 1
            chunk_size = max(1, seq_len // num_chunks)
            actual_num_chunks = (seq_len + chunk_size - 1) // chunk_size  # Ceiling division
            
            # Pad the sequence to be a multiple of chunk_size for easier reshaping
            pad_size = actual_num_chunks * chunk_size - seq_len
            if pad_size > 0:
                xs_padded = jnp.pad(xs_, ((0, pad_size), (0, 0)), mode='edge')
                ts_padded = jnp.pad(ts_, ((0, pad_size),), mode='edge')
            else:
                xs_padded = xs_
                ts_padded = ts_
            
            # Reshape to (num_chunks, chunk_size, feature_dim)
            xs_chunks = xs_padded.reshape(actual_num_chunks, chunk_size, -1)
            ts_chunks = ts_padded.reshape(actual_num_chunks, chunk_size)
            
            # Define scan function for the forward pass within each chunk
            def process_chunk(carry, chunk_data):
                theta_state, x_prev, t_prev = carry
                chunk_xs, chunk_ts = chunk_data
                
                # Inner scan function for processing steps within a chunk
                def f(carry, input_signal):
                    thet, x_prev, t_prev = carry
                    x_true, t_curr = input_signal
                    
                    A = self.As[0]
                    B = self.Bs[0]
                    x_t = x_true
                    
                    if self.time_as_channel:
                        x_t = jnp.concatenate([t_curr, x_t], axis=-1)
                        x_prev = jnp.concatenate([t_prev, x_prev], axis=-1)
                    
                    thet_next = A @ thet + B @ (x_t - x_prev)  # Key step
                    
                    if self.weights_lim is not None:
                        thet_next = jnp.clip(thet_next, -self.weights_lim, self.weights_lim)
                    
                    return (thet_next, x_true, t_curr), (thet_next,)
                    
                # Process the current chunk
                (final_theta, final_x, final_t), (theta_outs,) = jax.lax.scan(
                    f, (theta_state, x_prev, t_prev), (chunk_xs, chunk_ts[:, None])
                )
                
                # Return final state and outputs for this chunk
                return (final_theta, final_x, final_t), theta_outs
            
            # Initialize the theta state
            if self.init_state_layers is None:
                theta_init = self.thetas_init[0]
            else:
                theta_init = self.thetas_init[0](xs_[0])
                
            if self.noise_theta_init is not None:
                theta_init += jax.random.normal(k_, theta_init.shape) * self.noise_theta_init
            
            # Initial carry for the first chunk
            initial_carry = (theta_init, xs_[0], ts_[0:1])
            
            # Process all chunks sequentially
            _, theta_outs_chunks = jax.lax.scan(
                process_chunk, initial_carry, (xs_chunks, ts_chunks)
            )
            
            # Flatten the outputs and trim to original sequence length
            theta_outs = theta_outs_chunks.reshape(-1, theta_outs_chunks.shape[-1])[:seq_len]
            
            # Apply theta to get predictions
            @eqx.filter_vmap
            def apply_theta(theta, t_curr, x_curr):
                delta_t = ts_[1] - ts_[0]
                root_utils = self.root_utils[0]
                shapes, treedef, static, *_ = root_utils
                params = unflatten_pytree(theta, shapes, treedef)
                root_fun = eqx.combine(params, static)
                
                root_in = t_curr + delta_t
                if self.input_prev_data:
                    root_in = jnp.concatenate([root_in, x_curr], axis=-1)
                    
                if not self.classification:
                    x_next = root_fun(root_in, self.std_lower_bound, self.dtanh_params)
                else:
                    x_next = root_fun(root_in)
                    
                return x_next
            
            xs_hat = apply_theta(theta_outs, ts_[:, None], xs_)
            return xs_hat
        
        # Batched version of the forward pass
        ks = jax.random.split(k, xs.shape[0])
        return eqx.filter_vmap(forward_tbptt)(xs, ts, ks)














class GRU(eqx.Module):
    """ Gated Recurrent Unit """
    cell: eqx.Module
    decoder: eqx.Module

    data_size: int
    time_as_channel: bool
    forcing_prob: float
    time_as_channel: bool

    predict_uncertainty: bool
    std_lower_bound: float

    def __init__(self, 
                 data_size, 
                 hidden_size, 
                 predict_uncertainty=True,
                 time_as_channel=True,
                 forcing_prob=1.0,
                 std_lower_bound=None,
                 key=None):

        self.time_as_channel = time_as_channel
        input_dim = 1+data_size if time_as_channel else data_size

        self.cell = eqx.nn.GRUCell(input_dim, hidden_size, use_bias=True, key=key)
        self.decoder = eqx.nn.Linear(hidden_size, 2*data_size if predict_uncertainty else data_size, use_bias=True, key=key)

        self.data_size = data_size
        self.time_as_channel = time_as_channel
        self.forcing_prob = forcing_prob

        self.predict_uncertainty = predict_uncertainty
        self.std_lower_bound = std_lower_bound

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
                h, x_hat, x_prev, t_prev = carry
                x_true, t_curr, key = input_signal

                if inference_start is not None:
                    x_t = jnp.where(t_curr<=inference_start/ts_.shape[0], x_true, x_hat)
                else:
                    x_t = jnp.where(jax.random.bernoulli(key, self.forcing_prob), x_true, x_hat)

                if self.time_as_channel:
                    x_t = jnp.concatenate([t_curr, x_t], axis=-1)
                    x_prev = jnp.concatenate([t_prev, x_prev], axis=-1)

                h_next = self.cell(x_t, h)
                x_next = self.decoder(h_next)

                x_next_mean = x_next[:x_true.shape[0]]
                if self.predict_uncertainty:
                    x_next_std = jnp.clip(x_next[x_true.shape[0]:], self.std_lower_bound, None)
                    x_next = jnp.concatenate([x_next_mean, x_next_std], axis=-1)

                return (h_next, x_next_mean, x_hat, t_curr), (x_next, )

            keys = jax.random.split(k_, xs_.shape[0])

            _, (xs_hat, ) = jax.lax.scan(f, (jnp.zeros(self.cell.hidden_size), xs_[0], xs_[0], ts_[0:1]), (xs_, ts_[:, None], keys))

            return xs_hat

        ## Batched version of the forward pass
        ks = jax.random.split(k, xs.shape[0])
        return eqx.filter_vmap(forward)(xs, ts, ks)






class LSTM(eqx.Module):
    """ Gated Recurrent Unit """
    cell: eqx.Module
    decoder: eqx.Module

    data_size: int
    time_as_channel: bool
    forcing_prob: float
    time_as_channel: bool

    predict_uncertainty: bool
    std_lower_bound: float

    def __init__(self, 
                 data_size, 
                 hidden_size, 
                 predict_uncertainty=True,
                 time_as_channel=True,
                 forcing_prob=1.0,
                 std_lower_bound=None,
                 key=None):

        self.time_as_channel = time_as_channel
        input_dim = 1+data_size if time_as_channel else data_size

        self.cell = eqx.nn.LSTMCell(input_dim, hidden_size, use_bias=True, key=key)
        self.decoder = eqx.nn.Linear(hidden_size, 2*data_size if predict_uncertainty else data_size, use_bias=True, key=key)

        self.data_size = data_size
        self.time_as_channel = time_as_channel
        self.forcing_prob = forcing_prob

        self.predict_uncertainty = predict_uncertainty
        self.std_lower_bound = std_lower_bound

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
                hc, x_hat, x_prev, t_prev = carry
                x_true, t_curr, key = input_signal

                if inference_start is not None:
                    x_t = jnp.where(t_curr<=inference_start/ts_.shape[0], x_true, x_hat)
                else:
                    x_t = jnp.where(jax.random.bernoulli(key, self.forcing_prob), x_true, x_hat)

                if self.time_as_channel:
                    x_t = jnp.concatenate([t_curr, x_t], axis=-1)
                    x_prev = jnp.concatenate([t_prev, x_prev], axis=-1)

                hc_next = self.cell(x_t, hc)
                x_next = self.decoder(hc_next[0])

                x_next_mean = x_next[:x_true.shape[0]]
                if self.predict_uncertainty:
                    x_next_std = jnp.clip(x_next[x_true.shape[0]:], self.std_lower_bound, None)
                    x_next = jnp.concatenate([x_next_mean, x_next_std], axis=-1)

                return (hc_next, x_next_mean, x_hat, t_curr), (x_next, )

            keys = jax.random.split(k_, xs_.shape[0])

            _, (xs_hat, ) = jax.lax.scan(f, ((jnp.zeros(self.cell.hidden_size), jnp.zeros(self.cell.hidden_size)), xs_[0], xs_[0], ts_[0:1]), (xs_, ts_[:, None], keys))

            return xs_hat

        ## Batched version of the forward pass
        ks = jax.random.split(k, xs.shape[0])
        return eqx.filter_vmap(forward)(xs, ts, ks)




def make_model(key, data_size, nb_classes, config):
    """ Make a model using the given key and kwargs """

    model_type = config['model']['model_type']

    if model_type == "wsm":
        model_args = {
            "data_size": data_size,
            "width_size": config['model']['root_width_size'],
            "depth": config['model']['root_depth'],
            "activation": config['model']['root_activation'],
            "final_activation": config['model']['root_final_activation'],
            "nb_classes": nb_classes,
            "init_state_layers": config['model']['init_state_layers'],
            "input_prev_data": config['model']['input_prev_data'],
            "predict_uncertainty": config['training']['use_nll_loss'],
            "time_as_channel": config['model']['time_as_channel'],
            "forcing_prob": config['model']['forcing_prob'],
            "std_lower_bound": config['model']['std_lower_bound'],
            "weights_lim": config['model']['weights_lim'],
            "noise_theta_init": config['model']['noise_theta_init'],
            "autoregressive_train": config['training']['autoregressive'],
        }
        model = WSM(key=key, **model_args)
        if not isinstance(model.thetas_init[0], GradualMLP):
            print(f"Number of weights in the root network: {count_params((model.thetas_init,))/1000:3.1f} k")
        else:
            print(f"Number of weights in the root network: {model.thetas_init[0].layers[-1].out_features/1000:3.1f} k")
            print(f"Number of learnable parameters in the initial hyper-network: {count_params((model.thetas_init,))/1000:3.1f} k")
        print(f"Number of learnable parameters for the recurrent transition: {count_params((model.As, model.Bs))/1000:3.1f} k")
    elif model_type == "wsm-lstm":
        raise NotImplementedError("LSTM transition model not implemented yet")
    elif model_type == "wsm-gru":
        raise NotImplementedError("GRU transition model not implemented yet")
    elif model_type == "rnn":
        raise NotImplementedError("Standard RNN not implemented yet")
    elif model_type == "lstm":
        model_args = {
            "data_size": data_size,
            "predict_uncertainty": config['training']['use_nll_loss'],
            "time_as_channel": config['model']['time_as_channel'],
            "forcing_prob": config['model']['forcing_prob'],
            "std_lower_bound": config['model']['std_lower_bound'],
            "hidden_size": config['model']['rnn_hidden_size'],
        }
        model = LSTM(key=key, **model_args)
    elif model_type == "gru":
        model_args = {
            "data_size": data_size,
            "predict_uncertainty": config['training']['use_nll_loss'],
            "time_as_channel": config['model']['time_as_channel'],
            "forcing_prob": config['model']['forcing_prob'],
            "std_lower_bound": config['model']['std_lower_bound'],
            "hidden_size": config['model']['rnn_hidden_size'],
        }
        model = GRU(key=key, **model_args)
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
        "dynamic_tanh_init": False,
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


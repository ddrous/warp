#%%
from utils import flatten_pytree, unflatten_pytree, count_params, compute_kernel

import jax
import jax.numpy as jnp
import equinox as eqx







##################### WARP (OR "WSM" FOR WEIGHT-SPACE MODEL) MODEL DEFINITION(S) ########################

class RootMLP(eqx.Module):
    """ Root network \theta: t -> x_t, whose weights are the latent space of the WSM """
    network: eqx.Module
    props: any              ## Properties of the root network
    predict_uncertainty: bool
    final_activation: any

    def __init__(self, 
                 input_dim,
                 output_dim, 
                 width_size, 
                 depth, 
                 activation=jax.nn.relu,
                 final_activation=jax.nn.tanh,
                 predict_uncertainty=True,           ## Predict the std in addition to the mean
                 key=None):

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
            # print("Output shape before split: ", out.shape, flush=True)
            mean, std = jnp.split(out, 2, axis=-1)

            if self.final_activation is not None:
                mean = self.final_activation(mean)
            elif dtanh is not None:
                a, b, alpha, beta = dtanh
                mean = alpha*jnp.tanh((mean - b) / a) + beta
            else:
                pass

            # std = jnp.clip(std, -4, -3)
            # std = jnp.exp(std)

            std = jax.nn.softplus(std)
            if std_lb is not None:
                std = jnp.clip(std, std_lb, None)

            return jnp.concatenate([mean, std], axis=-1)






class RootMLP_Classif(eqx.Module):
    """ Root network \theta: t -> y_t, whose weights are the WSM hidden space for classification """
    network: eqx.Module
    props: any              ## Properties of the root network

    def __init__(self, 
                 nb_classes, 
                 width_size, 
                 depth, 
                 activation=jax.nn.relu,
                 positional_enc_dim=0,               ## Dimension of the positional encoding to be added to the input
                 key=None):

        input_dim = 1 + positional_enc_dim
        output_dim = nb_classes

        self.network = eqx.nn.MLP(input_dim, output_dim, width_size, depth, activation, key=key)
        self.props = (input_dim, output_dim, width_size, depth, activation)

    def __call__(self, tx):
        out = self.network(tx)
        return out



class GradualMLP(eqx.Module):
    """ Gradual MLP, with input neurons gradually increasing to output_dim """
    layers: list

    def __init__(self, input_dim, output_dim, hidden_layers=2, activation=jax.nn.tanh, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)

        if hidden_layers == 2:
            ## one thirds an two thirds
            hidden_size1 = int(2/3*input_dim + 1/3*output_dim)
            hidden_size2 = int(1/3*input_dim + 2/3*output_dim)
            in_layer = eqx.nn.Linear(input_dim, hidden_size1, key=keys[0])
            hidden_layer = eqx.nn.Linear(hidden_size1, hidden_size2, key=keys[1])
            out_layer = eqx.nn.Linear(hidden_size2, output_dim, key=keys[2])
            self.layers = [in_layer, activation, hidden_layer, activation, out_layer]

        elif hidden_layers == 1:
            ## half
            hidden_size = int(0.5*input_dim + 0.5*output_dim)
            print("All variables: input_dim, hidden_size, output_dim: ", input_dim, hidden_size, output_dim, flush=True)
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


class ConvNet1D(eqx.Module):
    """ Convolutional network for 1D data, e.g. time series """
    layers: eqx.Module
    activation: any

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, n_layers=2, activation=jax.nn.relu, key=None):
        keys = jax.random.split(key, n_layers)
        self.layers = []
        self.activation = activation
        self.layers.append(eqx.nn.Conv1d(in_channels, hidden_channels, kernel_size, padding="SAME", key=keys[0]))

        for i in range(1, n_layers):
            self.layers.append(eqx.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding="SAME", key=keys[i]))

        self.layers.append(eqx.nn.Conv1d(hidden_channels, out_channels, kernel_size, padding="SAME", key=keys[-1]))

    def __call__(self, x):
        """ Forward pass of the convolutional network
            x: (batch, in_channels, time)
            Returns: (batch, out_channels, time)
        """
        y = x
        for layer in self.layers:
            y = layer(y)
            if self.activation is not None:
                y = self.activation(y)
        return y










class WSM(eqx.Module):
    """ WARP: Weight Space Seq2Seq Model, with RNN transition function """
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
    stochastic_ar: bool
    smooth_inference: bool

    conv_embedding: eqx.Module
    conv_de_embedding: eqx.Module

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
                 noise_theta_init=None,             ## Potential noise to be added to the initial theta 
                 nb_wsm_layers=1,                   ## TODO to be used as in Fig 2. of https://arxiv.org/abs/2202.07022
                 autoregressive_train=True,
                 stochastic_ar=True,
                 smooth_inference=None,             ## Whether to use smooth inference (i.e. no reparametrization trick at inference)
                 positional_encoding=None,              ## Positional encoding to be added to the root network input
                 preferred_output_dim=None,              ## Output dimension to be used by the root network (e.g. for classification, in-context learning, etc.)
                 conv_embedding=None,               ## Convolutional embedding properties to be applied to the input sequences
                 key=None):

        keys = jax.random.split(key, num=nb_wsm_layers)
        builtin_fns = {"relu":jax.nn.relu, "tanh":jax.nn.tanh, 'softplus':jax.nn.softplus, 'swish':jax.nn.swish, "identity": lambda x: x, "sin": jnp.sin, "cos": jnp.cos}
        thetas_init = []
        root_utils = []
        As = []
        Bs = []

        if conv_embedding is not None:
            out_chans, kernel_size = conv_embedding[0], conv_embedding[1]
            print("Using convolutional embedding with out_chans: ", out_chans, " and kernel_size: ", kernel_size)
            # self.conv_embedding = eqx.nn.Conv1d(in_channels=data_size,
            #                                     out_channels=out_chans,
            #                                     kernel_size=kernel_size,
            #                                     padding="SAME",
            #                                     # padding=0,
            #                                     use_bias=False,
            #                                     key=keys[0])
            # seq_length = 32

            ## Define the padding depending on whether causal or not
            causal_conv = False if len(conv_embedding) < 3 else (conv_embedding[2] == 1)
            if causal_conv:
                print("Using causal convolution")
                padding = ((kernel_size-1, 0),)  ## Causal padding
            else:
                print("Using non-causal convolution")
                padding = "SAME"                 ## Non-causal padding

            self.conv_embedding = eqx.nn.Conv1d(in_channels=data_size,
                                                out_channels=out_chans,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                use_bias=False,
                                                key=keys[0])
            ## Initialise the kernel at 1s to get an initial cumsum
            # print("THid id ythe conv enbedding kernel: ", self.conv_embedding)
            # self.conv_embedding = eqx.tree_at(lambda x: x.weight, self.conv_embedding, jnp.ones_like(self.conv_embedding.weight))

            # self.conv_embedding = ConvNet1D(in_channels=data_size-1,
            #                                 out_channels=out_chans-1,
            #                                 hidden_channels=128,
            #                                 kernel_size=kernel_size,
            #                                 n_layers=1,
            #                                 activation=builtin_fns[activation],
            #                                 key=keys[0])

            self.conv_de_embedding = eqx.nn.Conv1d(in_channels=1,
                                                    out_channels=1,
                                                    kernel_size=3,
                                                    # padding='VALID', # No padding
                                                    padding='SAME', # No padding
                                                    use_bias=False,
                                                    key=key,)

            # self.conv_de_embedding = eqx.nn.Conv1d(in_channels=1,
            #                                         out_channels=1,
            #                                         kernel_size=kernel_size,
            #                                         padding=((kernel_size-1, 0),), # Causal padding
            #                                         use_bias=False,
            #                                         key=key,)
            # diff_kernel = jnp.array([[[-1., 1.]]])
            # self.conv_de_embedding = eqx.tree_at(lambda x: x.weight, self.conv_de_embedding, diff_kernel)

        else:
            self.conv_embedding = None
            self.conv_de_embedding = None

        positional_enc_dim = positional_encoding[0] if positional_encoding is not None else 0
        for i in range(nb_wsm_layers):
            if nb_classes is None:          ## Regression problem
                if isinstance(final_activation, str):
                    final_activation_fn = builtin_fns[final_activation]
                elif isinstance(final_activation, float):
                    final_activation_fn = lambda x: jnp.clip(x, -final_activation, final_activation)
                else:
                    final_activation_fn = None

                if conv_embedding is None:
                    input_dim = 1+positional_enc_dim+data_size if input_prev_data else 1+positional_enc_dim
                else:
                    input_dim = 1+positional_enc_dim+out_chans if input_prev_data else 1+positional_enc_dim
                    # input_dim = 1+positional_enc_dim+data_size-1 if input_prev_data else 1+positional_enc_dim
                output_dim = 2*data_size if predict_uncertainty else data_size

                input_dim = 3             #@TODO: just for Comphy data
                output_dim = 6            #@TODO: just for Comphy data

                ## Check if provided root_output dimension in yaml
                if preferred_output_dim is not None:
                    output_dim = preferred_output_dim
                    print("Using preferred output dimension: ", output_dim, flush=True)

                # print("\n\nUsing WSM layer ", i, " with input_dim: ", input_dim, " and output_dim: ", output_dim, flush=True)

                root = RootMLP(input_dim, 
                            output_dim,
                            width_size, 
                            depth, 
                            builtin_fns[activation], 
                            final_activation=final_activation_fn,
                            predict_uncertainty=predict_uncertainty,
                            key=keys[i])
            else:                           ## Classification problem
                root = RootMLP_Classif(nb_classes, 
                            width_size, 
                            depth, 
                            builtin_fns[activation], 
                            positional_enc_dim=positional_enc_dim,
                            key=keys[i])

            params, static = eqx.partition(root, eqx.is_array)
            weights, shapes, treedef = flatten_pytree(params)
            root_utils.append((shapes, treedef, static, root.props))

            if init_state_layers is None:
                thetas_init.append(weights)                              ## The latent space of the model
            else:
                theta_init_in_dim = data_size if conv_embedding is None else out_chans
                thetas_init.append(GradualMLP(theta_init_in_dim, weights.shape[0], init_state_layers, builtin_fns[activation], key=keys[0]))

            latent_size = weights.shape[0]
            print("Latent size of WSM layer ", i, ": ", latent_size, "input_dim: ", input_dim, "output_dim: ", output_dim, flush=True)
            As.append(jnp.eye(latent_size))                             ## The most stable matrix: identity

            if conv_embedding is None:
                B_in_dim = data_size+1 if time_as_channel else data_size
            else:
                B_in_dim = out_chans+1 if time_as_channel else out_chans
            print("B_in_dim: ", B_in_dim, " latent_size: ", latent_size, "data_size: ", data_size, flush=True)
            B = jnp.zeros((latent_size, B_in_dim))
            # B += jax.random.normal(keys[i], B.shape)*1e-3             ## TODO Initial perturbation to avoid getting stuck at 0 ?
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
        self.stochastic_ar = stochastic_ar
        self.smooth_inference = smooth_inference if smooth_inference is not None else (not stochastic_ar)

        if self.classification and self.ar_train:
            raise ValueError("The WSM model is not compatible with autoregressive training for classification tasks.")

        self.std_lower_bound = std_lower_bound

        if isinstance(final_activation, list):      ## and len(final_activation) == 4
            # self.dtanh_params = jnp.array(final_activation, dtype=jnp.float32)
            self.dtanh_params = jnp.array(final_activation)
        else:
            self.dtanh_params = None



    def __call__(self, xs, ts, k, inference_start=None):
        """ Forward pass of the model on batch of sequences
            xs: (batch, time, data_size)
            ts: (batch, time, dim)
            k:  (key_dim)
            inference_start: whether/when to use the model in autoregressive mode
            """

        # return self.conv_call(xs, ts, k)

        if self.classification:
            return self.non_ar_call(xs, ts, k)
        else:       ## Regression task
            if (self.ar_train) or (inference_start is not None):
                if self.stochastic_ar:
                    return self.ar_call_stochastic(xs, ts, k, inference_start=inference_start)
                else:
                    return self.ar_call(xs, ts, k, inference_start=inference_start)
            else:
                return self.non_ar_call(xs, ts, k)

    def ar_call(self, xs, ts, k, inference_start=None):
        """ Forward pass of the model on batch of sequences, ==autoregressively==
            xs: (batch, time, data_size)
            ts: (batch, time, dim)
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
                    x_t = jnp.where(t_curr[0]<=inference_start/ts_.shape[0], x_true, x_hat)
                else:
                    # key1, key2 = jax.random.split(key)
                    x_t = jnp.where(jax.random.bernoulli(key, self.forcing_prob), x_true, x_hat)

                if self.time_as_channel:
                    x_t = jnp.concatenate([x_t, t_curr[:1]], axis=-1)
                    x_prev = jnp.concatenate([x_prev, t_prev[:1]], axis=-1)

                thet_next = A@thet + B@(x_t - x_prev)     ## Key step

                if self.weights_lim is not None:
                    thet_next = jnp.clip(thet_next, -self.weights_lim, self.weights_lim)

                shapes, treedef, static, _ = root_utils
                params = unflatten_pytree(thet_next, shapes, treedef)
                root_fun = eqx.combine(params, static)

                # root_in = t_curr+delta_t
                # if self.input_prev_data:
                #     root_in = jnp.concatenate([root_in, x_prev[:x_true.shape[0]]], axis=-1)
                # x_next = root_fun(root_in, self.std_lower_bound, self.dtanh_params)                                  ## Evaluated at the next time step

                ###############################
                down_factor = 8
                H, W = 320//down_factor, 480//down_factor
                xs_coords = jnp.linspace(-1, 1, W)
                ys_coords = jnp.linspace(-1, 1, H)
                xs_grid, ys_grid = jnp.meshgrid(xs_coords, ys_coords)
                coords = jnp.stack([xs_grid, ys_grid], axis=-1).reshape(-1, 2)  ## (H*W, 2)

                ## Expand t_curr to match coords shape
                t_curr = jnp.repeat(t_curr[None, :]+delta_t, coords.shape[0], axis=0)   ## Shape (H*W, 1)
                root_in = jnp.concatenate([t_curr, coords], axis=-1)  ## Shape (H*W, 3)
                x_next = eqx.filter_vmap(root_fun, in_axes=(0, None, None))(root_in, self.std_lower_bound, self.dtanh_params)       ## Shape (H*W, 3)

                ## Split in two along the last dimension, and combine again to make sure
                x_next = jnp.concat([arr.flatten() for arr in jnp.split(x_next, 2, axis=-1)])
                ###############################

                x_next_mean = x_next[:x_true.shape[0]]

                # if inference_start is not None:           ## @TODO: Teacher-forced x_prev ?
                #     x_prev_next = jnp.where(t_curr<=inference_start/ts_.shape[0], x_true, x_hat)
                # else:
                #     x_prev_next = jnp.where(jax.random.bernoulli(key2, self.forcing_prob*1.0), x_true, x_hat)
                #     # x_prev_next = x_true
                # return (thet_next, x_next_mean, x_prev_next, t_curr), (x_next, )

                return (thet_next, x_next_mean, x_hat, t_curr), (x_next, )
                # return (thet_next, x_next_mean, x_true, t_curr), (x_next, )

            if self.init_state_layers is None:
                theta_init = self.thetas_init[0]
            else:
                theta_init = self.thetas_init[0](xs_[0])

            if self.noise_theta_init is not None:
                theta_init += jax.random.normal(k_, theta_init.shape)*self.noise_theta_init

            keys = jax.random.split(k_, xs_.shape[0])

            _, (xs_hat, ) = jax.lax.scan(f, (theta_init, xs_[0], xs_[0], ts_[0]), (xs_, ts_, keys))

            return xs_hat

        ## Batched version of the forward pass
        ks = jax.random.split(k, xs.shape[0])
        return eqx.filter_vmap(forward)(xs, ts, ks)

    def ar_call_stochastic(self, xs, ts, k, inference_start=None):
        """ Forward pass of the model on batch of sequences, ==autoregressively and stochastically==
            xs: (batch, time, data_size)
            ts: (batch, time)
            k:  (key_dim)
            inference_start: whether/when to use the model in autoregressive mode
            """

        def forward(xs_, ts_, k_):
            """ Forward pass on a single sequence """

            def f(carry, input_signal):
                thet, x_mu_sigma, x_prev, t_prev = carry
                x_true, t_curr, key = input_signal
                delta_t = t_curr - t_prev

                x_hat_mean, x_hat_std = jnp.split(x_mu_sigma, 2, axis=-1)

                A = self.As[0]
                B = self.Bs[0]
                root_utils = self.root_utils[0]

                if inference_start is not None:
                    if self.smooth_inference:
                        x_hat = x_hat_mean
                    else:
                        x_hat = jax.random.normal(key, x_hat_mean.shape)*x_hat_std + x_hat_mean
                    # x_t = jnp.where(t_curr<=inference_start/ts_.shape[0], x_true, x_hat)
                    x_t = jnp.where(t_curr[0]<=inference_start/ts_.shape[0], x_true, x_hat)
                    # print("Shape of things first: ", B.shape, x_hat_mean.shape, x_t.shape, "\n\n\n\n")
                else:
                    x_hat = jax.random.normal(key, x_hat_mean.shape)*x_hat_std + x_hat_mean
                    x_t = jnp.where(jax.random.bernoulli(key, self.forcing_prob), x_true, x_hat)        ## Sample x_hat from the normal distribution

                if self.time_as_channel:
                    x_t = jnp.concatenate([x_t, t_curr[:1]], axis=-1)
                    x_prev = jnp.concatenate([x_prev, t_prev[:1]], axis=-1)

                thet_next = A@thet + B@(x_t - x_prev)     ## Key step

                if self.weights_lim is not None:
                    thet_next = jnp.clip(thet_next, -self.weights_lim, self.weights_lim)

                shapes, treedef, static, _ = root_utils
                params = unflatten_pytree(thet_next, shapes, treedef)
                root_fun = eqx.combine(params, static)
                root_in = t_curr+delta_t
                # root_in = t_curr
                # root_in = t_curr[0:1]
                # print("root_in shape: ", root_in.shape, "x_prev shape: ", x_prev.shape, "x_true shape: ", x_true.shape, "\n\n\n\n")
                # root_in = jnp.array([0.0])

                # if self.input_prev_data:
                #     root_in = jnp.concatenate([root_in, x_prev[:x_true.shape[0]]], axis=-1)
                # x_next = root_fun(root_in, self.std_lower_bound, self.dtanh_params)                                  ## Evaluated at the next time step

                ###############################
                down_factor = 8
                H, W = 320//down_factor, 480//down_factor
                xs_coords = jnp.linspace(-1, 1, W)
                ys_coords = jnp.linspace(-1, 1, H)
                xs_grid, ys_grid = jnp.meshgrid(xs_coords, ys_coords)
                coords = jnp.stack([xs_grid, ys_grid], axis=-1).reshape(-1, 2)  ## (H*W, 2)

                ## Expand t_curr to match coords shape
                t_curr_s = jnp.repeat(t_curr[None, :]+delta_t, coords.shape[0], axis=0)   ## Shape (H*W, 1)
                root_in = jnp.concatenate([t_curr_s, coords], axis=-1)  ## Shape (H*W, 3)
                x_next = eqx.filter_vmap(root_fun, in_axes=(0, None, None))(root_in, self.std_lower_bound, self.dtanh_params)       ## Shape (H*W, 3)

                ## Split in two along the last dimension, and combine again to make sure
                x_next = jnp.concat([arr.flatten() for arr in jnp.split(x_next, 2, axis=-1)])
                ###############################

                return (thet_next, x_next, x_hat, t_curr), (x_next, )

            if self.init_state_layers is None:
                theta_init = self.thetas_init[0]
            else:
                theta_init = self.thetas_init[0](xs_[0])

            if self.noise_theta_init is not None:
                theta_init += jax.random.normal(k_, theta_init.shape)*self.noise_theta_init

            keys = jax.random.split(k_, xs_.shape[0])
            mu_sigma_init = jnp.concatenate([xs_[0], jnp.zeros_like(xs_[0])], axis=-1)        ## Initial mean and std

            # print("The shape of things is: \n\n", xs_.shape, ts_.shape, mu_sigma_init.shape, k_.shape)
            # _, (xs_hat, ) = jax.lax.scan(f, (theta_init, mu_sigma_init, xs_[0], ts_[0:1]), (xs_, ts_[:, None], keys))
            _, (xs_hat, ) = jax.lax.scan(f, (theta_init, mu_sigma_init, xs_[0], ts_[0]), (xs_, ts_, keys))

            return xs_hat

        ## Batched version of the forward pass
        ks = jax.random.split(k, xs.shape[0])
        return eqx.filter_vmap(forward)(xs, ts, ks)

    def non_ar_call(self, xs, ts, k):
        """ Forward pass of the model on batch of sequences, ==recurrent, but non-autoregressively==
            xs: (batch, time, data_size)
            ts: (batch, time, dim)
            k:  (key_dim)
            """

        def forward(xs_, ts_, k_):
            """ Forward pass on a single sequence """

            ## Apply a convolution to xs_ here (channel first)
            if self.conv_embedding is not None:
                # last_feat = xs_[:, -1:]  ## Keep the last feature (e.g. y_t) as it is
                # other_feats = xs_[:, :-1]  ## Keep all other features (e.g. x_t) to be transformed
                # xs_ = self.conv_embedding(other_feats.T).T
                # xs_ = jnp.concatenate([xs_, last_feat], axis=-1)  ## Add the last feature back to the input signal

                # xs_ = self.conv_embedding(xs_.T).T
                # print("\n\nShape of xs_ before conv embedding: ", xs_.shape)
                # xs_ = self.conv_embedding(xs_.T).T

                ## VEctorise the convolutional embedding and apply along channels
                # xs_ = eqx.filter_vmap(self.conv_embedding)(xs_.T[:, None, :])[:, 0, :].T  ## Apply the convolutional embedding to the input signal
                # xs_ = eqx.filter_vmap(jax.lax.stop_gradient(self.conv_embedding))(xs_.T[:, None, :])[:, 0, :].T
                # xs_ = eqx.filter_vmap(self.conv_embedding)(xs_.T[:, None, :])[:, 0, :].T
                xs_ = self.conv_embedding(xs_.T).T

                # print("\n\nShape of xs_ after conv embedding: ", xs_.shape, flush=True)
                # xs_ = xs_[:ts.shape[0]]  ## Apply the convolution to the input signal, and keep only the first ts.shape[0] time steps

            def f(carry, input_signal):
                thet, x_prev, t_prev = carry
                x_true, t_curr = input_signal

                A = self.As[0]
                B = self.Bs[0]

                x_t = x_true
                if self.time_as_channel:
                    x_t = jnp.concatenate([t_curr[:1], x_t], axis=-1)
                    x_prev = jnp.concatenate([t_prev[:1], x_prev], axis=-1)

                thet_next = A@thet + B@(x_t - x_prev)     ## Key step
                # thet_next = A@thet + B@(x_t-x_prev) + jnp.sum(x_t*x_prev)
                # thet_next = (jnp.eye(thet.shape[0]) - t_curr[0]**2)@thet + t_curr[0]*B@(x_t - x_prev)     ## Key step

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

            _, (theta_outs, ) = jax.lax.scan(f, (theta_init, xs_[0], ts_[0]), (xs_, ts_))

            #################################################################################
            # @eqx.filter_vmap
            # def apply_theta(theta, t_curr, x_curr):
            #     # delta_t = ts_[1] - ts_[0]
            #     root_utils = self.root_utils[0]
            #     shapes, treedef, static, _ = root_utils
            #     params = unflatten_pytree(theta, shapes, treedef)
            #     root_fun = eqx.combine(params, static)
            #     # root_in = t_curr+delta_t
            #     root_in = t_curr

            #     if self.input_prev_data:
            #         root_in = jnp.concatenate([root_in, x_curr], axis=-1)

            #     if not self.classification:
            #         x_next = root_fun(root_in, self.std_lower_bound, self.dtanh_params)                                                 ## Evaluated at the next time step
            #     else:
            #         x_next = root_fun(root_in)                                      ## Evaluated at the next time step
            #     return x_next

            # xs_hat = apply_theta(theta_outs, ts_, xs_)
            #################################################


            ####### In some cases, we might apply theta to video; in which case theta takes x,y coords as input ###
            ## Define the x,y coords here, with Hight, and Weight = 320, 480
            down_factor = 8
            H, W = 320//down_factor, 480//down_factor
            xs_coords = jnp.linspace(-1, 1, W)
            ys_coords = jnp.linspace(-1, 1, H)
            xs_grid, ys_grid = jnp.meshgrid(xs_coords, ys_coords)
            coords = jnp.stack([xs_grid, ys_grid], axis=-1).reshape(-1, 2)  ## (H*W, 2)

            @eqx.filter_vmap
            def apply_theta(theta, t_curr):
                root_utils = self.root_utils[0]
                shapes, treedef, static, _ = root_utils
                params = unflatten_pytree(theta, shapes, treedef)
                root_fun = eqx.combine(params, static)

                ## Expand t_curr to match coords shape
                t_curr = jnp.repeat(t_curr[None, :], coords.shape[0], axis=0)   ## Shape (H*W, 1)

                root_in = jnp.concatenate([t_curr, coords], axis=-1)  ## Shape (H*W, 3)

                x_next = eqx.filter_vmap(root_fun)(root_in)       ## Shape (H*W, 3)

                # Rehaspe to (H, W, 3)
                x_next = x_next.reshape(H, W, -1)
                return x_next.flatten()

                # ## Split in two along the last dimension, and combine again to make sure
                # x_next = jnp.concat([arr.flatten() for arr in jnp.split(x_next, 2, axis=-1)])
                # return x_next

            xs_hat = apply_theta(theta_outs, ts_)    ## (time, H*W*3)
            #################################################

            ## Apply the de-embedding convolution if needed
            if self.conv_de_embedding is not None:
                xs_hat = eqx.filter_vmap(self.conv_de_embedding)(xs_hat.T[:, None, :])[:, 0, :].T
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
                        x_t = jnp.concatenate([t_curr[:1], x_t], axis=-1)
                        x_prev = jnp.concatenate([t_prev[:1], x_prev], axis=-1)
                    
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

    def conv_call(self, xs, ts, k):
        """ Forward pass of the model on batch of sequences, ==convolutional==
            xs: (batch, time, data_size)
            ts: (batch, time)
            k:  (key_dim)
            """

        fft = False
        seq_len = xs.shape[1]
        input_signal = xs[:, 1:] - xs[:, :-1]  ## Calculate the difference between consecutive time steps

        ##### THEORETICALLY GROUNDED APPROACH FOR COMPUTING \DELTA X_0 ? #####
        # ## Call the scan function
        # if self.init_state_layers is None:
        #     theta_init = self.thetas_init[0]
        # else:
        #     theta_init = self.thetas_init[0](xs_[0])

        # if self.noise_theta_init is not None:
        #     theta_init += jax.random.normal(k_, theta_init.shape)*self.noise_theta_init

        input_signal = jnp.concatenate([xs[:, 0:1], input_signal], axis=1)  ## Add the first time step back to the input signal
        K_conv = compute_kernel(self.As[0], self.Bs[0], seq_len)  ## Compute the convolution kernel: (time, output_size, data_size)

        if not fft:
            ## Apply the Jax.lax convolution
            thetas = jax.lax.conv_general_dilated(lhs=input_signal.transpose(0, 2, 1),             ## (batch, data_size, time)
                                                    rhs=K_conv.transpose(1, 2, 0),          ## (output_size, data_size, time)
                                                    window_strides=(1,), 
                                                    padding="SAME").transpose(0, 2, 1)  ## Reshape to original (batch, time, output_size)
        else:
            ## Use the convolution theorem
            input_fft = jnp.fft.fft(input_signal[:, :, None, :], axis=1)                    ## (batch, time, 1, data_size)
            K_conv_fft = jnp.fft.fft(K_conv[None, :, :, :], axis=1)                          ## (1, time, out_size, data_size)
            theta_fft = jnp.sum(K_conv_fft * input_fft, axis=-1)                             ## (batch, time, out_size)
            thetas = jnp.fft.ifft(theta_fft, axis=1).real

        @eqx.filter_vmap
        @eqx.filter_vmap
        def apply_theta(theta, t_curr, x_curr):
            delta_t = ts[0, 1] - ts[0, 0]  ## Assuming uniform time steps
            ## Applying theta for a single time step
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

        print("SHAPES", thetas.shape, ts.shape, xs.shape)
        xs_hat = apply_theta(thetas, ts[...,None], xs)

        return xs_hat







##################### GRU MODEL DEFINITION ########################


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
                    x_t = jnp.where(t_curr[0]<=inference_start/ts_.shape[0], x_true, x_hat)
                else:
                    # x_t = x_true
                    x_t = jnp.where(jax.random.bernoulli(key, self.forcing_prob), x_true, x_hat)

                if self.time_as_channel:
                    x_t = jnp.concatenate([t_curr[:1], x_t], axis=-1)
                    x_prev = jnp.concatenate([t_prev[:1], x_prev], axis=-1)

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




##################### LSTM MODEL DEFINITION ########################


class LSTM(eqx.Module):
    """ Long Short-Term Memory """
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
                    x_t = jnp.where(t_curr[0]<=inference_start/ts_.shape[0], x_true, x_hat)
                else:
                    x_t = jnp.where(jax.random.bernoulli(key, self.forcing_prob), x_true, x_hat)

                if self.time_as_channel:
                    x_t = jnp.concatenate([t_curr[:1], x_t], axis=-1)
                    x_prev = jnp.concatenate([t_prev[:1], x_prev], axis=-1)

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





##################### FEED-FORWARD MODEL DEFINITION ########################

class FFNN(eqx.Module):
    """ Feed-Forward Neural Network """
    data_size: int
    sequence_length: bool
    layers: list

    def __init__(self, data_size, key=None):
        """ Initialize the model """

        self.data_size = data_size
        self.sequence_length = 699

        num_layers = 3
        width_size = 1024
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                in_features = self.sequence_length
            else:
                in_features = width_size

            if i == num_layers-1:
                out_features = self.sequence_length
            else:
                out_features = width_size

            key, subkey = jax.random.split(key)
            layer = eqx.nn.Linear(in_features, out_features, use_bias=True, key=key)
            self.layers.append(layer)

    def __call__(self, xs, ts, k, inference_start=None):
        """ Forward pass of the model on batch 
            """

        def forward(x):
            """ Forward pass on a single sequence """
            for layer in self.layers[:-1]:
                x = layer(x)
                x = jax.nn.relu(x)
            x = self.layers[-1](x)
            return x[...,None]

        return eqx.filter_vmap(forward)(xs.squeeze(-1)) 






##################### PUTTING THINGS TOGETHER ########################

def make_model(key, data_size, nb_classes, config, logger):
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
            "stochastic_ar": config['training']['stochastic'],
            "positional_encoding": config['model'].get('positional_encoding', False),
            "preferred_output_dim": config['model'].get('root_output_dim', None),
            "conv_embedding": config['model'].get('conv_embedding', None),
        }

        if 'smooth_inference' in config['training']:
            model_args['smooth_inference'] = config['training']['smooth_inference']

        if model_args['autoregressive_train'] and model_args['stochastic_ar'] and not config['training']['use_nll_loss']:
            raise ValueError("The WSM model is not compatible with stochastic autoregressive training without NLL loss.")

        if not model_args['smooth_inference'] and not model_args['stochastic_ar']:
            raise ValueError("Non-smooth (stochastic, reparametrization trick) inference cannot be used without stochastic training.")

        model = WSM(key=key, **model_args)
        if not isinstance(model.thetas_init[0], GradualMLP):
            logger.info(f"Number of weights in the root network:  {count_params((model.thetas_init,))/1000:3.1f} k")
        else:
            logger.info(f"Number of weights in the root network: {model.thetas_init[0].layers[-1].out_features/1000:3.1f} k")
            logger.info(f"Number of learnable parameters in the initial hyper-network: {count_params((model.thetas_init,))/1000:3.1f} k")

        logger.info(f"Number of learnable parameters for the recurrent transition: {count_params((model.As, model.Bs))/1000:3.1f} k")


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


    elif model_type == "ffnn":
        model_args = {
            "data_size": data_size,
            # "sequence_length": config['model']['sequence_length'],
        }
        model = FFNN(key=key, **model_args)

    logger.info(f"Number of learnable parameters in the model: {count_params(model)/1000:3.1f} k")

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


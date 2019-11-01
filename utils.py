import math
import numpy as np
from scipy.stats import norm
from matplotlib.colors import Normalize
import tensorflow as tf
from tensorflow.contrib.distributions import bijectors, softplus_inverse
# from tensorflow.contrib.keras import layers
# from tensorflow.contrib.keras import models
from edward import KLqp
from edward.models import RandomVariable
from edward.util import get_session, get_variables, Progbar, transform
import six
import os
from datetime import datetime
# from tf_gbds.layers import PKBiasLayer, PKRowBiasLayer

layers = tf.keras.layers
models = tf.keras.models


class set_cbar_zero(Normalize):
    """set_cbar_zero(midpoint = float)       default: midpoint = 0.
    Normalizes and sets the center of any colormap to the desired value which
    is set using midpoint.
    """
    def __init__(self, vmin=None, vmax=None, midpoint=0., clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = ([min(self.vmin, -self.vmax), self.midpoint,
                 max(self.vmax, -self.vmin)], [0, 0.5, 1])
        return np.ma.masked_array(np.interp(value, x, y))


def gauss_convolve(x, sigma, pad_method="edge_pad"):
    """Smoothing with gaussian convolution
    Pad Methods:
        * edge_pad: pad with the values on the edges
        * extrapolate: extrapolate the end pad based on dx at the end
        * zero_pad: pad with zeros
    """
    method_types = ["edge_pad", "extrapolate", "zero_pad"]
    if pad_method not in method_types:
        raise Exception("Pad method not recognized")
    edge = int(math.ceil(5 * sigma))
    fltr = norm.pdf(range(-edge, edge), loc=0, scale=sigma)
    fltr = fltr / sum(fltr)

    szx = x.size

    if pad_method == "edge_pad":
        buff = np.ones(edge)
        xx = np.append((buff * x[0]), x)
        xx = np.append(xx, (buff * x[-1]))
    elif pad_method == "extrapolate":
        buff = np.ones(edge)
        # linear extrapolation for end edge buffer
        end_dx = x[-1] - x[-2]
        end_buff = np.cumsum(end_dx * np.ones(edge)) + x[-1]
        xx = np.append((buff * x[0]), x)
        xx = np.append(xx, end_buff)
    else:
        # zero pad
        buff = np.zeros(edge)
        xx = np.append(buff, x)
        xx = np.append(xx, buff)

    y = np.convolve(xx, fltr, mode="valid")
    y = y[:szx]
    return y


def smooth_trial(trial, sigma=4.0, pad_method="extrapolate"):
    """Apply Gaussian convolution Smoothing method to real data
    """
    rtrial = trial.copy()
    for i in range(rtrial.shape[1]):
        rtrial[:, i] = gauss_convolve(rtrial[:, i], sigma,
                                      pad_method=pad_method)
    return rtrial


def load_data(data_dir, hps):
    """ Load data from given directory
    """
    features = {"trajectory": tf.FixedLenFeature((), tf.string)}
    if hps.extra_conds:
        features.update({"extra_conds": tf.FixedLenFeature(
            (), tf.string)})

    def _read_data(example):
        parsed_features = tf.parse_single_example(example, features)

        trajectory = tf.reshape(
            tf.decode_raw(parsed_features["trajectory"], tf.float32),
            # [-1, hps.obs_dim])
            [-1, hps.obs_dim * 2])
        data = (trajectory,)

        if "extra_conds" in parsed_features:
            extra_conds = tf.reshape(
                tf.decode_raw(parsed_features["extra_conds"], tf.float32),
                [-1, hps.extra_dim])
            data += (extra_conds,)

        return data

    with tf.name_scope("preprocessing"):
        dataset = tf.data.TFRecordDataset(data_dir)
        dataset = dataset.map(_read_data)
        dataset = dataset.shuffle(
            buffer_size=1000000, seed=tf.random_uniform(
                [], minval=-2**63+1, maxval=2**63-1, dtype=tf.int64))
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(hps.B))
        # if hps.B > 1:
        #     dataset = dataset.map(_pad_data)
        iterator = dataset.make_initializable_iterator("iterator")

    return iterator


# def get_max_velocities(datasets, dim):
#     """Get the maximium velocities from datasets
#     """
#     max_vel = [[] for _ in range(dim)]
#     for d in range(len(datasets)):
#         for i in range(len(datasets[d])):
#             for c in range(dim):
#                 if np.abs(np.diff(datasets[d][i][:, c])).max() > 0.001:
#                     max_vel[c].append(
#                         np.abs(np.diff(datasets[d][i][:, c])).max())

#     return np.array([max(vel) for vel in max_vel], np.float32)


def get_max_velocities(data_dirs, dim, clip):
    """Get the maximium velocities from datasets
    """
    max_vel = np.zeros((dim), np.float32)
    n_trials = []

    feature = {"trajectory": tf.FixedLenFeature((), tf.string)}

    def _get_trial(example):
        data_dict = tf.parse_single_example(example, feature)
        traj = tf.reshape(
            tf.decode_raw(data_dict["trajectory"], tf.float32), [-1, dim])

        return traj

    for data_dir in data_dirs:
        dataset = tf.data.TFRecordDataset(data_dir)
        dataset = dataset.map(_get_trial)
        traj = dataset.make_one_shot_iterator().get_next()
        trial_max_vel = tf.reduce_max(tf.abs(traj[1:] - traj[:-1]), 0,
                                      name="trial_maximum_velocity")

        dataset_size = 0
        with tf.Session() as sess:
            while True:
                try:
                    max_vel = np.maximum(sess.run(trial_max_vel), max_vel)
                    dataset_size += 1
                except tf.errors.OutOfRangeError:
                    break
        n_trials.append(dataset_size)

    if clip:
        return max_vel, n_trials
    else:
        return np.around(max_vel, decimals=3) + 0.001, n_trials


def get_vel(traj, max_vel):
    """Input a time series of positions and include velocities for each
    coordinate in each row
    """
    with tf.name_scope("get_velocity"):
        vel = tf.pad(
            tf.divide(traj[:, 1:] - traj[:, :-1], max_vel.astype(np.float32),
                      name="standardize"), [[0, 0], [1, 0], [0, 0]],
            name="pad_zero")
        states = tf.concat([traj, vel], -1, name="states")

        return states


def get_accel(traj, max_vel):
    """Input a time series of positions and include velocities and acceleration
    for each coordinate in each row
    """
    with tf.name_scope("get_acceleration"):
        states = get_vel(traj, max_vel)
        accel = traj[:, 2:] - 2 * traj[1:-1] + traj[:-2]
        accel = tf.pad(accel, [[0, 0], [2, 0], [0, 0]], name="pad_zeros")
        states = tf.concat([states, accel], -1, name="states")

        return states


def get_network(name, input_dim, output_dim, hidden_dim, num_layers):
    """Returns a NN with the specified parameters.
    """

    with tf.variable_scope(name):
        M = models.Sequential(name=name)
        M.add(layers.InputLayer(input_shape=(None, input_dim), name="Input"))

        for i in range(num_layers):
            if i == num_layers - 1:
                M.add(layers.Dense(
                    output_dim, "linear",
                    kernel_initializer=tf.random_normal_initializer(
                        stddev=0.1), name="Dense_%s" % (i + 1)))
            else:
                M.add(layers.Dense(
                    hidden_dim, "relu",
                    # kernel_initializer=tf.orthogonal_initializer(),
                    kernel_initializer=tf.glorot_normal_initializer(),
                    name="Dense_%s" % (i + 1)))

        return M


def get_model_params(game_name, agent_name, agent_col, agent_dim, state_dim,
                     extra_dim, K_1, K_2, gen_n_layers, gen_hidden_dim,
                     goal_boundaries, goal_boundary_penalty,
                     goal_precision_penalty, rec_lag, lstm_units,
                     rec_n_layers, rec_hidden_dim, penalty_Q,
                     unc_epsilon, epsilon_trainable, epsilon_penalty,
                     init_temperature, epoch):

    with tf.variable_scope("model_parameters"):
        with tf.variable_scope(agent_name):
            states = layers.Input((None, state_dim), name="states")
            extra_conds = layers.Input(
                (None, extra_dim), name="extra_conditions")
            NN_inputs = layers.Concatenate(
                axis=-1, name="NN_inputs")([states, extra_conds])

            GMM_NN_outputs = layers.Dense(
                K_2 * agent_dim * 2, "linear",
                kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                name="GMM_NN_Dense")(NN_inputs)
            GMM_mu = layers.Reshape((-1, K_2, agent_dim), name="mu")(
                layers.Lambda(lambda x: x[..., :(K_2 * agent_dim)])(
                    GMM_NN_outputs))
            lambda_floor = 4e2
            GMM_lambda = layers.Reshape((-1, K_2, agent_dim), name="lambda")(
                layers.Lambda(lambda x: tf.nn.softplus(
                    x[..., (K_2 * agent_dim):]) + lambda_floor)(
                    GMM_NN_outputs))

            exit_NN_FCLayers = get_network(
                "exit_NN_FCLayers", state_dim + extra_dim, K_1 * K_2 * 2,
                gen_hidden_dim, gen_n_layers)
            exit_probs = layers.Softmax(name="exit_probs")(
                layers.Reshape((-1, K_1, K_2, 2), name="exit_logits")(
                    exit_NN_FCLayers(NN_inputs)))

            # z_1_init_NN_FCLayers = get_network(
            #     "z_1_init_NN_FCLayers", state_dim + extra_dim, K_1,
            #     gen_hidden_dim, gen_n_layers)
            # z_1_init_probs = layers.Softmax(name="z_1_init_probs")(
            #     z_1_init_NN_FCLayers(NN_inputs))
            z_1_init_logits = tf.Variable(
                np.log(np.ones((K_1), np.float32) / K_1),
                name="z_1_initial_logits")

            z_2_NN_outputs = layers.Dense(
                K_1 * K_2 * K_2, "linear",
                kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                name="z_2_NN_Dense")(NN_inputs)
            z_2_probs = layers.Softmax(name="z_2_probs")(
                layers.Reshape((-1, K_1, K_2, K_2), name="z_2_logits")(
                    z_2_NN_outputs))

            p_NN = models.Model(
                inputs=[states, extra_conds],
                # outputs=[GMM_mu, GMM_lambda, exit_probs, z_1_init_probs,
                #          z_2_probs],
                outputs=[GMM_mu, GMM_lambda, exit_probs, z_2_probs],
                name="p_NN")

            if epsilon_trainable:
                unc_eps_init = tf.Variable(
                    unc_epsilon * np.ones((1, agent_dim), np.float32),
                    name="unc_eps")
                eps_pen = tf.to_float(tf.minimum(
                    epsilon_penalty * (10. ** ((epoch - 1) / 10)), 1e5),
                    "epsilon_penalty")
            else:
                unc_eps_init = tf.constant(
                    unc_epsilon * np.ones((1, agent_dim), np.float32),
                    name="unc_eps")
                eps_pen = None

            fix_ep = 10
            unc_Kp = tf.Variable(tf.multiply(
                softplus_inverse(1.), tf.ones(agent_dim, tf.float32),
                "unc_Kp_init"), name="unc_Kp")
            Kp_ = tf.nn.softplus(unc_Kp)
            Kp = tf.cond(tf.greater(epoch, fix_ep),
                         lambda: Kp_, lambda: tf.stop_gradient(Kp_))

            # temperature_gen = tf.maximum(init_temperature * (.95 ** (
            #     tf.maximum(epoch - fix_ep, 0) / 5)), .1, "temperature")
            temperature_gen = init_temperature

        with tf.variable_scope("posterior"):
            padded_traj = layers.Input(
                (None, agent_dim * (rec_lag + 1)),
                name="trajectory_with_lag")
            q_extra_conds = layers.Input(
                (None, extra_dim), name="extra_conditions")
            q_NN_inputs = layers.Concatenate(axis=-1, name="q_NN_inputs")(
                [padded_traj, q_extra_conds])
            q_NN_LSTM = layers.Bidirectional(
                layers.LSTM(lstm_units, return_sequences=True),
                name="q_LSTM")(q_NN_inputs)
            q_NN_FCLayers = get_network(
                "q_NN_FCLayers", lstm_units * 2,
                agent_dim * 2 + K_1 + K_2 + 2, rec_hidden_dim, rec_n_layers)
            q_NN_outputs = q_NN_FCLayers(q_NN_LSTM)
            qg_mu = layers.Lambda(
                lambda x: x[..., :agent_dim], name="qg_mu")(q_NN_outputs)
            qg_lambda = layers.Lambda(
                lambda x: tf.nn.softplus(x[..., agent_dim:(agent_dim * 2)]),
                name="qg_lambda")(q_NN_outputs)
            qz_1_probs = layers.Softmax(name="qz_1_probs")(layers.Lambda(
                lambda x: x[..., (agent_dim * 2):(agent_dim * 2 + K_1)],
                name="qz_1_logits")(q_NN_outputs))
            qz_2_probs = layers.Softmax(name="qz_2_probs")(layers.Lambda(
                lambda x: x[..., (agent_dim * 2 + K_1):(
                    agent_dim * 2 + K_1 + K_2)],
                name="qz_2_logits")(q_NN_outputs))
            exit_probs = layers.Softmax(name="exit_probs")(layers.Lambda(
                lambda x: x[..., -2:], name="exit_logits")(q_NN_outputs))
            q_NN = models.Model(
                inputs=[padded_traj, q_extra_conds],
                outputs=[qg_mu, qg_lambda, qz_1_probs, qz_2_probs,
                         exit_probs], name="q_NN")

            temperature_rec = init_temperature

            # with tf.name_scope("penalty_Q"):
            #     if penalty_Q is not None:
            #         q_params["p"] = penalty_Q

        params = dict(
            game_name=game_name, agent_name=agent_name, agent_col=agent_col,
            agent_dim=agent_dim, state_dim=state_dim, extra_dim=extra_dim,
            # K_1=K_1, K_2=K_2, p_NN=p_NN,
            K_1=K_1, K_2=K_2, p_NN=p_NN, z_1_init_logits=z_1_init_logits,
            g_bounds=goal_boundaries, g_bounds_pen=goal_boundary_penalty,
            g_prec_pen=goal_precision_penalty, unc_eps=unc_eps_init,
            eps_trainable=epsilon_trainable, eps_pen=eps_pen,
            unc_Kp=unc_Kp, Kp=Kp, t_p=temperature_gen,
            lag=rec_lag, q_NN=q_NN, t_q=temperature_rec)

        return params


# def get_rec_params(agent_dim, extra_dim, output_dim, lag, n_layers,
#                    hidden_dim, penalty_Q=None, name="recognition"):
#     """Return a dictionary of timeseries-specific parameters for recognition
#        model
#     """

#     with tf.variable_scope(name):
#         Mu_net = get_network(
#             "Mu_NN", (agent_dim * (lag + 1) + extra_dim), output_dim,
#             hidden_dim, n_layers)
#         Lambda_net = get_network(
#             "Lambda_NN", agent_dim * (lag + 1) + extra_dim, output_dim ** 2,
#             hidden_dim, n_layers)
#         LambdaX_net = get_network(
#             "LambdaX_NN", agent_dim * (lag + 1) + extra_dim, output_dim ** 2,
#             hidden_dim, n_layers)

#         dyn_params = dict(
#             A=tf.Variable(
#                 .9 * np.eye(output_dim), name="A", dtype=tf.float32),
#             QinvChol=tf.Variable(
#                 np.eye(output_dim), name="QinvChol", dtype=tf.float32),
#             Q0invChol=tf.Variable(
#                 np.eye(output_dim), name="Q0invChol", dtype=tf.float32))

#         trainable_variables = (Mu_net.variables + Lambda_net.variables +
#                                LambdaX_net.variables +
#                                list(dyn_params.values()))

#         rec_params = dict(
#             dyn_params=dyn_params, NN_Mu=Mu_net, NN_Lambda=Lambda_net,
#             NN_LambdaX=LambdaX_net, trainable_variables=trainable_variables,
#             lag=lag, extra_dim=extra_dim)

#         with tf.name_scope("penalty_Q"):
#             if penalty_Q is not None:
#                 rec_params["p"] = penalty_Q

#         return rec_params


# def get_PID(dim, epoch):
#     fix_ep = 40
#     with tf.variable_scope("PID"):
#         unc_Kp = tf.Variable(tf.multiply(
#             softplus_inverse(1.), tf.ones(dim, tf.float32), "unc_Kp_init"),
#                              name="unc_Kp")
#         unc_Ki = tf.Variable(tf.multiply(
#             softplus_inverse(1e-6), tf.ones(dim, tf.float32), "unc_Ki_init"),
#                              name="unc_Ki")
#         unc_Kd = tf.Variable(tf.multiply(
#             softplus_inverse(1e-6), tf.ones(dim, tf.float32), "unc_Kd_init"),
#                              name="unc_Kd")

#         Kp = tf.nn.softplus(unc_Kp, "Kp")
#         Ki = tf.nn.softplus(unc_Ki, "Ki")
#         Kd = tf.nn.softplus(unc_Kd, "Kd")

#         PID = {}
#         PID["vars"] = [unc_Kp] + [unc_Ki] + [unc_Kd]
#         PID["Kp"] = tf.cond(tf.greater(epoch, fix_ep),
#                             lambda: Kp, lambda: tf.stop_gradient(Kp))
#         PID["Ki"] = tf.cond(tf.greater(epoch, fix_ep),
#                             lambda: Ki, lambda: tf.stop_gradient(Ki))
#         PID["Kd"] = tf.cond(tf.greater(epoch, fix_ep),
#                             lambda: Kd, lambda: tf.stop_gradient(Kd))
#         # PID["Kp"] = Kp
#         # PID["Ki"] = Ki
#         # PID["Kd"] = Kd

#         return PID

# def get_PID(dim):
#     with tf.variable_scope("PID"):
#         PID = {}
#         PID["unc_Kp"] = tf.Variable(
#             tf.multiply(softplus_inverse(1.), tf.ones(dim, tf.float32),
#                         "unc_Kp_init"), name="unc_Kp")
#         PID["unc_Ki"] = tf.Variable(
#             tf.multiply(softplus_inverse(1e-6), tf.ones(dim, tf.float32),
#                         "unc_Ki_init"), name="unc_Ki")
#         PID["unc_Kd"] = tf.Variable(
#             tf.multiply(softplus_inverse(1e-6), tf.ones(dim, tf.float32),
#                         "unc_Kd_init"), name="unc_Kd")

#     return PID


# def pad_batch(arrays, mode="edge"):
#     max_len = np.max([len(a) for a in arrays])
#     if mode == "edge":
#         return np.array([np.pad(a, ((0, max_len - len(a)), (0, 0)),
#                                 "edge") for a in arrays])
#     elif mode == "zero":
#         return np.array(
#             [np.pad(a, ((0, max_len - len(a)), (0, 0)), "constant",
#                     constant_values=0) for a in arrays])


def pad_batch(batch, mode="edge"):
    max_len = tf.reduce_max(
        tf.map_fn(lambda x: tf.shape(x)[0], batch, dtype=tf.int32,
                  name="trial_length"), name="max_length")

    if mode == "edge":
        return tf.map_fn(
            lambda x: tf.concat(
                [x, tf.tile(tf.expand_dims(x[-1], 0),
                            [max_len - tf.shape(x)[0], 1])], 0), batch)
    elif mode == "zero":
        return tf.map_fn(
            lambda x: tf.pad(x, [[0, max_len - tf.shape(x)[0]], [0, 0]],
                             "constant"), batch)


def add_summary(summary_op, inference, session, feed_dict, step):
    if inference.n_print != 0:
        if step == 1 or step % inference.n_print == 0:
            summary = session.run(summary_op, feed_dict=feed_dict)
            inference.train_writer.add_summary(summary, step)


class KLqp_profile(KLqp):
    def __init__(self, options=None, run_metadata=None, latent_vars=None,
                 data=None):
        super(KLqp_profile, self).__init__(latent_vars=latent_vars, data=data)
        self.options = options
        self.run_metadata = run_metadata

    def update(self, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}

        for key, value in six.iteritems(self.data):
            if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
                feed_dict[key] = value

        sess = get_session()
        _, t, loss = sess.run([self.train, self.increment_t, self.loss],
                              options=self.options,
                              run_metadata=self.run_metadata,
                              feed_dict=feed_dict)

        if self.debug:
            sess.run(self.op_check, feed_dict)

        if self.logging and self.n_print != 0:
            if t == 1 or t % self.n_print == 0:
                summary = sess.run(self.summarize, feed_dict)
                self.train_writer.add_summary(summary, t)

        return {"t": t, "loss": loss}


class KLqp_clipgrads(KLqp):
    def __init__(self, *args, **kwargs):
        super(KLqp_clipgrads, self).__init__(*args, **kwargs)

    def initialize(self, n_iter=1000, n_print=None, scale=None,
                   auto_transform=True, logdir=None, log_timestamp=True,
                   log_vars=None, debug=False, optimizer=None, var_list=None,
                   use_prettytensor=False, global_step=None, n_samples=1,
                   kl_scaling=None, maxnorm=5.):

        if kl_scaling is None:
            kl_scaling = {}
        if n_samples <= 0:
            raise ValueError(
                "n_samples should be greater than zero: {}".format(n_samples))

        self.n_samples = n_samples
        self.kl_scaling = kl_scaling

        # from inference.py
        self.n_iter = n_iter
        if n_print is None:
            self.n_print = int(n_iter / 100)
        else:
            self.n_print = n_print

        self.progbar = Progbar(self.n_iter)
        self.t = tf.Variable(0, trainable=False, name="iteration")
        self.increment_t = self.t.assign_add(1)

        if scale is None:
            scale = {}
        elif not isinstance(scale, dict):
            raise TypeError("scale must be a dict object.")
        self.scale = scale

        self.transformations = {}
        if auto_transform:
            latent_vars = self.latent_vars.copy()
            self.latent_vars = {}
            self.latent_vars_unconstrained = {}
            for z, qz in six.iteritems(latent_vars):
                if hasattr(z, 'support') and hasattr(qz, 'support') and \
                        z.support != qz.support and qz.support != 'point':

                    z_unconstrained = transform(z)
                    self.transformations[z] = z_unconstrained

                    if qz.support == "points":
                        qz_unconstrained = qz
                    else:
                        qz_unconstrained = transform(qz)
                    self.latent_vars_unconstrained[
                        z_unconstrained] = qz_unconstrained

                    if z_unconstrained != z:
                        qz_constrained = transform(
                            qz_unconstrained,
                            bijectors.Invert(z_unconstrained.bijector))

                        try:
                            qz_constrained.params = \
                                    z_unconstrained.bijector.inverse(
                                        qz_unconstrained.params)
                        except:
                            pass
                    else:
                        qz_constrained = qz_unconstrained

                    self.latent_vars[z] = qz_constrained
                else:
                    self.latent_vars[z] = qz
                    self.latent_vars_unconstrained[z] = qz
            del latent_vars

        if logdir is not None:
            self.logging = True
            if log_timestamp:
                logdir = os.path.expanduser(logdir)
                logdir = os.path.join(
                    logdir,
                    datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S"))

            self._summary_key = tf.get_default_graph().unique_name(
                "summaries")
            self._set_log_variables(log_vars)
            self.train_writer = tf.summary.FileWriter(
                logdir, tf.get_default_graph())
        else:
            self.logging = False

        self.debug = debug
        if self.debug:
            self.op_check = tf.add_check_numerics_ops()

        self.reset = [tf.variables_initializer([self.t])]

        # from variational_inference.py
        if var_list is None:
            var_list = set()
            trainables = tf.trainable_variables()
            for z, qz in six.iteritems(self.latent_vars):
                var_list.update(get_variables(z, collection=trainables))
                var_list.update(get_variables(qz, collection=trainables))

            for x, qx in six.iteritems(self.data):
                if isinstance(x, RandomVariable) and \
                        not isinstance(qx, RandomVariable):
                    var_list.update(get_variables(x, collection=trainables))

        var_list = list(var_list)

        self.loss, grads_and_vars = self.build_loss_and_gradients(var_list)

        clipped_grads_and_vars = []
        for grad, var in grads_and_vars:
            if "kernel" in var.name or "bias" in var.name:
                clipped_grads_and_vars.append(
                    (tf.clip_by_norm(grad, maxnorm, axes=[0]), var))
            else:
                clipped_grads_and_vars.append((grad, var))
        # for grad, var in grads_and_vars:
        #     clipped_grads_and_vars.append(
        #         (tf.clip_by_value(grad, -1000., 1000.), var))
        del grads_and_vars

        if self.logging:
            tf.summary.scalar(
                "loss", self.loss, collections=[self._summary_key])
        for grad, var in clipped_grads_and_vars:
            tf.summary.histogram("gradient/" +
                                 var.name.replace(':', '/'),
                                 grad, collections=[self._summary_key])
            tf.summary.scalar("gradient_norm/" +
                              var.name.replace(':', '/'),
                              tf.norm(grad), collections=[self._summary_key])

        self.summarize = tf.summary.merge_all(key=self._summary_key)

        if optimizer is None and global_step is None:
            global_step = tf.Variable(0, trainable=False, name="global_step")

        if isinstance(global_step, tf.Variable):
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate, global_step, 100, 0.9, staircase=True)
        else:
            learning_rate = 0.01

        # Build optimizer.
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif isinstance(optimizer, str):
            if optimizer == 'gradientdescent':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            elif optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
            elif optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif optimizer == 'ftrl':
                optimizer = tf.train.FtrlOptimizer(learning_rate)
            elif optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
            else:
                raise ValueError('Optimizer class not found:', optimizer)
        elif not isinstance(optimizer, tf.train.Optimizer):
            raise TypeError(
                "Optimizer must be str, tf.train.Optimizer, or None.")

        with tf.variable_scope(None, default_name="optimizer") as scope:
            if not use_prettytensor:
                self.train = optimizer.apply_gradients(
                    clipped_grads_and_vars, global_step=global_step)
            else:
                import prettytensor as pt
                self.train = pt.apply_optimizer(
                    optimizer, losses=[self.loss],
                    global_step=global_step, var_list=var_list)

        self.reset.append(tf.variables_initializer(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)))

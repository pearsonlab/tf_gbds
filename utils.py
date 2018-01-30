import tensorflow as tf
import math
from os.path import expanduser
import h5py
from scipy.stats import norm
import numpy as np
import pandas as pd
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import constraints, models
from matplotlib.colors import Normalize
import edward as ed
from edward.models import Exponential, Gamma, PointMass
import six
from tf_gbds.layers import PKBiasLayer, PKRowBiasLayer


class set_cbar_zero(Normalize):
    """set_cbar_zero(midpoint = float)       default: midpoint = 0.
    Normalizes and sets the center of any colormap to the desired value which
    is set using midpoint.
    """
    def __init__(self, vmin=None, vmax=None, midpoint=0., clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = ([min(self.vmin, -self.vmax), self.midpoint, max(self.vmax,
                                                                -self.vmin)],
                [0, 0.5, 1])
        return np.ma.masked_array(np.interp(value, x, y))


def gauss_convolve(x, sigma, pad_method='edge_pad'):
    """Smoothing with gaussian convolution
    Pad Methods:
        * edge_pad: pad with the values on the edges
        * extrapolate: extrapolate the end pad based on dx at the end
        * zero_pad: pad with zeros
    """
    method_types = ['edge_pad', 'extrapolate', 'zero_pad']
    if pad_method not in method_types:
        raise Exception("Pad method not recognized")
    edge = int(math.ceil(5 * sigma))
    fltr = norm.pdf(range(-edge, edge), loc=0, scale=sigma)
    fltr = fltr / sum(fltr)

    szx = x.size

    if pad_method == 'edge_pad':
        buff = np.ones(edge)
        xx = np.append((buff * x[0]), x)
        xx = np.append(xx, (buff * x[-1]))
    elif pad_method == 'extrapolate':
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

    y = np.convolve(xx, fltr, mode='valid')
    y = y[:szx]
    return y


def smooth_trial(trial, sigma=4.0, pad_method='extrapolate'):
    """Apply Gaussian convolution Smoothing method to real data
    """
    rtrial = trial.copy()
    for i in range(rtrial.shape[1]):
        rtrial[:, i] = gauss_convolve(rtrial[:, i], sigma,
                                      pad_method=pad_method)
    return rtrial


# def gen_data(n_trials, n_obs, sigma=np.log1p(np.exp(-5. * np.ones((1, 2)))),
#              eps=np.log1p(np.exp(-10.)), Kp=1, Ki=0, Kd=0,
#              vel=1e-2 * np.ones((3))):
#     """Generate fake data to test the accuracy of the model
#     """
#     p = []
#     g = []

#     for _ in range(n_trials):
#         p_b = np.zeros((n_obs, 2), np.float32)
#         p_g = np.zeros((n_obs, 1), np.float32)
#         g_b = np.zeros((n_obs, 2), np.float32)
#         prev_error_b = 0
#         prev_error_g = 0
#         int_error_b = 0
#         int_error_g = 0

#         init_b_x = np.pi * (np.random.rand() * 2 - 1)
#         g_b_x_mu = 0.25 * np.sin(2. * (np.linspace(0, 2 * np.pi, n_obs) -
#                                        init_b_x))
#         init_b_y = np.pi * (np.random.rand() * 2 - 1)
#         g_b_y_mu = 0.25 * np.sin(2. * (np.linspace(0, 2 * np.pi, n_obs) -
#                                        init_b_y))
#         g_b_mu = np.hstack([g_b_x_mu.reshape(n_obs, 1),
#                             g_b_y_mu.reshape(n_obs, 1)])
#         g_b_lambda = np.array([16, 16], np.float32)
#         g_b[0] = g_b_mu[0]

#         for t in range(n_obs - 1):
#             g_b[t + 1] = ((g_b[t] + g_b_lambda * g_b_mu[t + 1]) /
#                           (1 + g_b_lambda))
#             var = sigma ** 2 / (1 + g_b_lambda)
#             g_b[t + 1] += (np.random.randn(1, 2) * np.sqrt(var)).reshape(2,)

#             error_b = g_b[t + 1] - p_b[t]
#             int_error_b += error_b
#             der_error_b = error_b - prev_error_b
#             u_b = (Kp * error_b + Ki * int_error_b + Kd * der_error_b +
#                    eps * np.random.randn(2,))
#             prev_error_b = error_b
#             p_b[t + 1] = p_b[t] + vel[1:] * np.clip(u_b, -1, 1)

#             error_g = p_b[t + 1, 1] - p_g[t]
#             int_error_g += error_g
#             der_error_g = error_g - prev_error_g
#             u_g = (Kp * error_g + Ki * int_error_g + Kd * der_error_g +
#                    eps * np.random.randn())
#             prev_error_g = error_g
#             p_g[t + 1] = p_g[t] + vel[0] * np.clip(u_g, -1, 1)

#         p.append(np.hstack((p_g, p_b)))
#         g.append(g_b)

#     return p, g


def gen_data(n_trials, n_obs, sigma=np.log1p(np.exp(-5. * np.ones((1, 3)))),
             eps=np.log1p(np.exp(-10.)), Kp=1, Ki=0, Kd=0,
             vel=1. * np.ones((3))):

    p = []
    g = []

    for _ in range(n_trials):
        p_b = np.zeros((n_obs, 2), np.float32)
        p_g = np.zeros((n_obs, 1), np.float32)
        g_b = np.zeros((n_obs, 2), np.float32)
        g_g = np.zeros((n_obs, 1), np.float32)
        prev_error_b = 0
        prev_error_g = 0
        prev2_error_b = 0
        prev2_error_g = 0
        u_b = 0
        u_g = 0

        init_b_x = np.pi * (np.random.rand() * 2 - 1)
        g_b_x_mu = (np.linspace(0, 0.975, n_obs) + 0.02 * np.sin(2. *
                    (np.linspace(0, 2 * np.pi, n_obs) - init_b_x)))

        init_b_y = np.pi * (np.random.rand() * 2 - 1)
        g_b_y_mu = (np.linspace(-0.2 + (np.random.rand() * 0.1 - 0.05),
                                0.4 + (np.random.rand() * 0.1 - 0.05),
                                n_obs) +
                    0.05 * np.sin(2. * (np.linspace(0, 2 * np.pi, n_obs) -
                                        init_b_y)))
        g_b_mu = np.hstack([g_b_x_mu.reshape(n_obs, 1),
                            g_b_y_mu.reshape(n_obs, 1)])
        g_b_lambda = np.array([1e4, 1e4])
        g_b[0] = g_b_mu[0]

        init_g = np.pi * (np.random.rand() * 2 - 1)
        g_g_mu = (np.linspace(-0.2 + (np.random.rand() * 0.1 - 0.05),
                              0.4 + (np.random.rand() * 0.1 - 0.05), n_obs) +
                  0.05 * np.sin(2. * (np.linspace(0, 2 * np.pi, n_obs) -
                                      init_g)))
        g_g_lambda = 1e4
        g_g[0] = g_g_mu[0]

        for t in range(n_obs - 1):
            g_b[t + 1] = ((g_b[t] + g_b_lambda * g_b_mu[t + 1]) /
                          (1 + g_b_lambda))
            var_b = sigma[0, 1:] ** 2 / (1 + g_b_lambda)
            g_b[t + 1] += (np.random.randn(1, 2) * np.sqrt(var_b)).reshape(2,)

            error_b = g_b[t + 1] - p_b[t]
            u_b += ((Kp + Ki + Kd) * error_b - (Kp + 2 * Kd) * prev_error_b +
                    Kd * prev2_error_b + eps * np.random.randn(2,))
            p_b[t + 1] = np.clip(p_b[t] + vel[1:] * np.clip(u_b, -1, 1),
                                 -1, 1)
            prev2_error_b = prev_error_b
            prev_error_b = error_b

            g_g[t + 1] = ((g_g[t] + g_g_lambda * g_g_mu[t + 1]) /
                          (1 + g_g_lambda))
            var_g = sigma[0, 0] ** 2 / (1 + g_g_lambda)
            g_g[t + 1] += np.random.randn() * np.sqrt(var_g)

            error_g = g_g[t + 1] - p_g[t]
            u_g += ((Kp + Ki + Kd) * error_g - (Kp + 2 * Kd) * prev_error_g +
                    Kd * prev2_error_g + eps * np.random.randn())
            p_g[t + 1] = np.clip(p_g[t] + vel[0] * np.clip(u_g, -1, 1), -1, 1)
            prev2_error_g = prev_error_g
            prev_error_g = error_g

        p.append(np.hstack((p_g, p_b)))
        g.append(np.hstack((g_g, g_b)))

    return p, g


def load_data(hps):
    """ Generate synthetic data set or load real data from local directory
    """
    train_data = []
    val_data = []
    if hps.synthetic_data:
        # data, goals = gen_data(
        #     n_trials=2000, n_obs=100, Kp=0.6, Ki=0.3, Kd=0.1)
        data, goals = gen_data(
            n_trials=2000, n_obs=100, Kp=0.5, Ki=0.2, Kd=0.1)
        np.random.seed(hps.seed)  # set seed for consistent train/val split
        val_goals = []
        for (trial_data, trial_goals) in zip(data, goals):
            if np.random.rand() <= hps.train_ratio:
                train_data.append(trial_data)
            else:
                val_data.append(trial_data)
                val_goals.append(trial_goals)
        np.save(hps.model_dir + "/train_data", train_data)
        np.save(hps.model_dir + "/val_data", val_data)
        np.save(hps.model_dir + "/val_goals", val_goals)
        train_conds = None
        val_conds = None
        train_ctrls = None
        val_ctrls = None
    elif hps.data_dir is not None:
        datafile = h5py.File(hps.data_dir, 'r')
        trajs = np.array(datafile["trajectories"], np.float32)
        if "conditions" in datafile:
            conds = np.array(datafile["conditions"], np.float32)
        else:
            conds = None
        if "control" in datafile:
            ctrls = np.array(datafile["control"], np.float32)
        else:
            ctrls = None
        datafile.close()
        np.random.seed(hps.seed)  # set seed for consistent train/val split
        train_ind = []
        val_ind = []
        np.random.seed(hps.seed)  # set seed for consistent train/val split
        for i in range(len(trajs)):
            if np.random.rand() <= hps.train_ratio:
                train_ind.append(i)
            else:
                val_ind.append(i)
        np.save(hps.model_dir + '/train_indices', train_ind)
        np.save(hps.model_dir + '/val_indices', val_ind)
        train_data = trajs[train_ind]
        val_data = trajs[val_ind]
        if conds is not None:
            train_conds = conds[train_ind]
            val_conds = conds[val_ind]
        else:
            train_conds = None
            val_conds = None
        if ctrls is not None:
            train_ctrls = ctrls[train_ind]
            val_ctrls = ctrls[val_ind]
        else:
            train_ctrls = None
            val_ctrls = None
    else:
        raise Exception("Data must be provided (either real or synthetic)")

    return train_data, train_conds, train_ctrls, val_data, val_conds, val_ctrls


def get_max_velocities(y_data, y_val_data):
    """Get the maximium velocities from data
    """
    goalie_y_vels = []
    ball_x_vels = []
    ball_y_vels = []
    for i in range(len(y_data)):
        if np.abs(np.diff(y_data[i][:, 0])).max() > 0.001:
            goalie_y_vels.append(np.abs(np.diff(y_data[i][:, 0])).max())
        if np.abs(np.diff(y_data[i][:, 1])).max() > 0.001:
            ball_x_vels.append(np.abs(np.diff(y_data[i][:, 1])).max())
        if np.abs(np.diff(y_data[i][:, 2])).max() > 0.001:
            ball_y_vels.append(np.abs(np.diff(y_data[i][:, 2])).max())
    for i in range(len(y_val_data)):
        if np.abs(np.diff(y_val_data[i][:, 0])).max() > 0.001:
            goalie_y_vels.append(np.abs(np.diff(y_val_data[i][:, 0])).max())
        if np.abs(np.diff(y_val_data[i][:, 1])).max() > 0.001:
            ball_x_vels.append(np.abs(np.diff(y_val_data[i][:, 1])).max())
        if np.abs(np.diff(y_val_data[i][:, 2])).max() > 0.001:
            ball_y_vels.append(np.abs(np.diff(y_val_data[i][:, 2])).max())

    return np.round(np.array([max(goalie_y_vels), max(ball_x_vels),
                              max(ball_y_vels)]), decimals=3)  # + 0.001


def get_vel(traj, max_vel):
    """Input a time series of positions and include velocities for each
    coordinate in each row
    """
    vel = tf.pad((traj[:, 1:] - traj[:, :-1]) / max_vel,
                 [[0, 0], [1, 0], [0, 0]], name='velocities')
    states = tf.concat([traj, vel], -1, name='states')
    return states


def get_accel(data, max_vel=None):
    """Input a time series of positions and include velocities and acceleration
    for each coordinate in each row
    """
    if isinstance(data, tf.Tensor):
        dims = tf.shape(data)[1]
    else:
        dims = data.shape[1]
    states = get_vel(data, max_vel=None)
    accel = data[2:] - 2 * data[1:-1] + data[:-2]
    accel = tf.concat([tf.zeros((2, dims), np.float32), accel], 0,
                      name='acceleration')
    states = tf.concat([states, accel], 1, name='position_vel_accel')
    return states


def get_agent_params(name, agent_dim, agent_cols, state_dim, extra_dim,
                     n_layers, hidden_dim, PKLparams, GMM_K, sigma,
                     boundaries_goal, penalty_goal, vel, epsilon,
                     penalty_ctrl, ctrl_residual_tolerance,
                     signal_clip, clip_range, clip_tol, eta,
                     penalty_ctrl_error):

    with tf.name_scope('%s_params' % name):
        NN, _ = get_network(
            'GMM_%s' % name, (state_dim + extra_dim),
            (GMM_K * agent_dim * 2 + GMM_K), hidden_dim, n_layers, PKLparams)
        g0_params = init_g0_params(agent_dim, GMM_K)
        agent_vel = vel[agent_cols]
        PID_params = get_PID_priors(agent_dim, agent_vel)

        params = dict(
            name=name, agent_dim=agent_dim, agent_cols=agent_cols,
            state_dim=state_dim, extra_dim=extra_dim, GMM_K=GMM_K, GMM_NN=NN,
            g0_params=g0_params, sigma=sigma, bounds_g=boundaries_goal,
            pen_g=penalty_goal, agent_vel=agent_vel, PID_params=PID_params,
            epsilon=epsilon, pen_u=penalty_ctrl,
            u_res_tol=ctrl_residual_tolerance, clip=signal_clip,
            clip_range=clip_range, clip_tol=clip_tol, eta=eta,
            pen_ctrl_error=penalty_ctrl_error)

        return params


def get_network(name, input_dim, output_dim, hidden_dim, num_layers,
                PKLparams=None, batchnorm=False, is_shooter=False,
                row_sparse=False, add_pklayers=False, filt_size=None):
    """Returns a NN with the specified parameters.
    Also returns a list of PKBias layers
    """

    with tf.variable_scope(name):
        M = models.Sequential()
        PKbias_layers = []
        M.add(layers.InputLayer(input_shape=(None, input_dim), name='Input'))
        if batchnorm:
            M.add(layers.BatchNormalization(name='BatchNorm'))
        if filt_size is not None:
            M.add(layers.ZeroPadding1D(padding=(filt_size - 1, 0),
                                       name='ZeroPadding'))
            M.add(layers.Conv1D(filters=hidden_dim, kernel_size=filt_size,
                                padding='valid', activation=tf.nn.relu,
                                name='Conv1D'))

        for i in range(num_layers):
            with tf.variable_scope('PK_Bias'):
                if is_shooter and add_pklayers:
                    if row_sparse:
                        PK_bias = PKRowBiasLayer(
                            M, PKLparams,
                            name='PKRowBias_%s' % (i + 1))
                    else:
                        PK_bias = PKBiasLayer(
                            M, PKLparams,
                            name='PKBias_%s' % (i + 1))
                    PKbias_layers.append(PK_bias)
                    M.add(PK_bias)

            if i == num_layers - 1:
                M.add(layers.Dense(
                    output_dim, activation='linear',
                    kernel_initializer=tf.random_normal_initializer(
                        stddev=0.1), name='Dense_%s' % (i + 1)))
            else:
                M.add(layers.Dense(
                    hidden_dim, activation='relu',
                    kernel_initializer=tf.orthogonal_initializer(),
                    name='Dense_%s' % (i + 1)))

        return M, PKbias_layers


def get_rec_params(obs_dim, extra_dim, agent_dim, lag, num_layers, hidden_dim,
                   penalty_Q=None, PKLparams=None, name='recognition'):
    """Return a dictionary of timeseries-specific parameters for recognition
       model
    """

    with tf.name_scope('%s_params' % name):
        mu_net, PKbias_layers_mu = get_network(
            'Mu_NN', (obs_dim * (lag + 1) + extra_dim), agent_dim,
            hidden_dim, num_layers, PKLparams)
        lambda_net, PKbias_layers_lambda = get_network(
            'Lambda_NN', obs_dim * (lag + 1) + extra_dim,
            agent_dim ** 2, hidden_dim, num_layers, PKLparams)
        lambdaX_net, PKbias_layers_lambdaX = get_network(
            'LambdaX_NN', obs_dim * (lag + 1) + extra_dim,
            agent_dim ** 2, hidden_dim, num_layers, PKLparams)

        with tf.name_scope('%s_Dyn_params'):
            Dyn_params = dict(
                A=tf.Variable(
                    .9 * np.eye(obs_dim), name='A', dtype=tf.float32),
                QinvChol=tf.Variable(
                    np.eye(obs_dim), name='QinvChol', dtype=tf.float32),
                Q0invChol=tf.Variable(
                    np.eye(obs_dim), name='Q0invChol', dtype=tf.float32))

        rec_params = dict(
            Dyn_params=Dyn_params,
            NN_Mu=dict(network=mu_net,
                       PKbias_layers=PKbias_layers_mu),
            NN_Lambda=dict(network=lambda_net,
                           PKbias_layers=PKbias_layers_lambda),
            NN_LambdaX=dict(network=lambdaX_net,
                            PKbias_layers=PKbias_layers_lambdaX),
            lag=lag)

        with tf.name_scope('penalty_Q'):
            if penalty_Q is not None:
                rec_params['p'] = penalty_Q

        return rec_params


# def get_gen_params_GBDS_GMM(obs_dim_agent, obs_dim, extra_dim, add_accel,
#                             yCols_agent, nlayers_gen, hidden_dim_gen,
#                             K, PKLparams, vel, sigma, eps,
#                             penalty_ctrl_error, boundaries_g, penalty_g,
#                             latent_u, clip, clip_range, clip_tol, eta, name):
#     """Return a dictionary of timeseries-specific parameters for generative
#        model
#     """

#     with tf.name_scope('get_states_%s' % name):
#         if add_accel:
#             get_states = get_accel
#             state_dim = obs_dim * 3
#         else:
#             get_states = get_vel
#             state_dim = obs_dim * 2

#     with tf.name_scope('gen_GMM_%s' % name):
#         GMM_net, _ = get_network('gen_GMM_%s' % name, (state_dim + extra_dim),
#                                  (K * obs_dim_agent * 2 + K),
#                                  hidden_dim_gen, nlayers_gen, PKLparams)

#     with tf.name_scope('PID_%s' % name):
#         PID_params = init_PID_params(name, obs_dim_agent, vel[yCols_agent])

#     with tf.name_scope('initial_goal_%s' % name):
#         g0_params = init_g0_params(name, obs_dim_agent, K)

#     gen_params = dict(all_vel=vel,
#                       vel=vel[yCols_agent],
#                       yCols=yCols_agent,  # which columns belong to the agent
#                       sigma=sigma,
#                       eps=eps,
#                       pen_ctrl_error=penalty_ctrl_error,
#                       bounds_g=boundaries_g,
#                       pen_g=penalty_g,
#                       latent_u=latent_u,
#                       clip=clip,
#                       clip_range=clip_range,
#                       clip_tol=clip_tol,
#                       eta=eta,
#                       get_states=get_states,
#                       PID_params=PID_params,
#                       g0_params=g0_params,
#                       GMM_net=GMM_net,
#                       GMM_k=K)

#     return gen_params


def get_PID_priors(dim, vel):
    """Return a dictionary of PID controller parameters
    """
    with tf.name_scope('PID_priors'):
        PID_priors = {}

        PID_priors['Kp'] = Gamma(
            np.ones(dim, np.float32) * 2, np.ones(dim, np.float32) * vel,
            name='Kp_prior', value=np.ones(dim, np.float32) / vel)
        PID_priors['Ki'] = Exponential(
            np.ones(dim, np.float32) / vel, name='Ki_prior',
            value=np.zeros(dim, np.float32))
        PID_priors['Kd'] = Exponential(
            np.ones(dim, np.float32), name='Kd_prior',
            value=np.zeros(dim, np.float32))

        return PID_priors


class Point_Mass(PointMass):
    def __init__(self, params, validate_args=True, allow_nan_stats=True,
                 name="PointMass"):
        super(Point_Mass, self).__init__(
            params=params, validate_args=validate_args,
            allow_nan_stats=allow_nan_stats, name=name)

    def _log_prob(self, value):
        return tf.zeros([])

    def _prob(self, value):
        return tf.zeros([])


def get_PID_posteriors(dim):
    with tf.name_scope('PID_posteriors'):
        PID_posteriors = {}

        unc_Kp_q = tf.Variable(tf.random_normal([dim]), name='unc_Kp_q')
        unc_Ki_q = tf.Variable(tf.random_normal([dim]), name='unc_Ki_q')
        unc_Kd_q = tf.Variable(tf.random_normal([dim]), name='unc_Kd_q')
        PID_posteriors['vars'] = ([unc_Kp_q] + [unc_Ki_q] + [unc_Kd_q])

        PID_posteriors['Kp'] = Point_Mass(
            params=tf.nn.softplus(unc_Kp_q), name='Kp_posterior')
        PID_posteriors['Ki'] = Point_Mass(
            params=tf.nn.softplus(unc_Ki_q), name='Ki_posterior')
        PID_posteriors['Kd'] = Point_Mass(
            params=tf.nn.softplus(unc_Kd_q), name='Kd_posterior')

        return PID_posteriors


def init_g0_params(dim, K):
    with tf.name_scope('g0_params'):
        g0_params = {}
        
        g0_params['K'] = K
        g0_params['mu'] = tf.Variable(tf.random_normal([K, dim]),
                                      dtype=tf.float32, name='mu')
        g0_params['unc_lambda'] = tf.Variable(
            tf.random_normal([K, dim]), dtype=tf.float32, name='unc_lambda')
        g0_params['unc_w'] = tf.Variable(
            tf.ones([K]), dtype=tf.float32, name='unc_w')

        return g0_params


def batch_generator(arrays, batch_size, randomize=True):
    """Minibatch generator over one dataset of shape
    (#observations, #dimensions)
    """
    n_trials = len(arrays)
    n_batch = math.floor(n_trials / batch_size)
    if randomize:
        np.random.shuffle(arrays)

    start = 0
    while True:
        batches = []
        for _ in range(n_batch):
            stop = start + batch_size
            diff = stop - n_trials

            if diff <= 0:
                batch = np.array(arrays[start:stop])
                start = stop
            batches.append(batch)

        yield batches


def batch_generator_pad(arrays, batch_size, extra_conds=None, ctrl_obs=None,
                        randomize=True):
    n_trials = len(arrays)
    n_batch = math.floor(n_trials / batch_size)
    if randomize:
        p = np.random.permutation(n_trials)
        arrays = arrays[p]
        if extra_conds is not None:
            extra_conds = extra_conds[p]
        if ctrl_obs is not None:
            ctrl_obs = ctrl_obs[p]

    start = 0
    while True:
        batches = []
        conds = []
        ctrls = []
        for _ in range(n_batch):
            stop = start + batch_size
            diff = stop - n_trials

            if diff <= 0:
                batch = arrays[start:stop]
                if extra_conds is not None:
                    cond = np.array(extra_conds[start:stop])
                if ctrl_obs is not None:
                    ctrl = np.array(ctrl_obs[start:stop])
                start = stop
            batch = pad_batch(batch)
            batches.append(batch)
            if extra_conds is not None:
                conds.append(cond)
            if ctrl_obs is not None:
                ctrl = pad_batch(ctrl, mode='zero')
                ctrls.append(ctrl)

        yield batches, conds, ctrls


def pad_batch(arrays, mode='edge'):
    max_len = np.max([len(a) for a in arrays])
    if mode == 'edge':
        return np.array([np.pad(a, ((0, max_len - len(a)), (0, 0)),
                                'edge') for a in arrays])
    elif mode == 'zero':
        return np.array([np.pad(a, ((0, max_len-len(a)), (0, 0)), 'constant',
                                constant_values=0) for a in arrays])


def pad_extra_conds(data, extra_conds=None):
    if extra_conds is not None:
        if not isinstance(extra_conds, tf.Tensor):
            extra_conds = tf.constant(extra_conds, dtype=tf.float32,
                                      name='extra_conds')
        extra_conds_repeat = tf.tile(
            tf.expand_dims(extra_conds, 1), [1, tf.shape(data)[1], 1],
            name='repeat_extra_conds')
        padded_data = tf.concat([data, extra_conds_repeat], axis=-1,
                                name='pad_extra_conds')
        return padded_data
    else:
        raise Exception('Must provide extra conditions.')


def add_summary(summary_op, inference, session, feed_dict, step):
    if inference.n_print != 0:
        if step == 1 or step % inference.n_print == 0:
            summary = session.run(summary_op, feed_dict=feed_dict)
            inference.train_writer.add_summary(summary, step)


class DatasetTrialIndexIterator(object):
    """Basic trial iterator
    """
    def __init__(self, y, randomize=False, batch_size=1):
        self.y = y
        self.randomize = randomize

    def __iter__(self):
        n_batches = len(self.y)
        if self.randomize:
            indices = list(range(n_batches))
            np.random.shuffle(indices)
            for i in indices:
                yield self.y[i]
        else:
            for i in range(n_batches):
                yield self.y[i]


class MultiDatasetTrialIndexIterator(object):
    """Trial iterator over multiple datasets of shape
    (ntrials, trial_len, trial_dimensions)
    """
    def __init__(self, data, randomize=False, batch_size=1):
        self.data = data
        self.randomize = randomize

    def __iter__(self):
        n_batches = len(self.data[0])
        if self.randomize:
            indices = list(range(n_batches))
            np.random.shuffle(indices)
            for i in indices:
                yield tuple(dset[i] for dset in self.data)
        else:
            for i in range(n_batches):
                yield tuple(dset[i] for dset in self.data)


class DataSetTrialIndexTF(object):
    """Tensor version of Minibatch iterator over one dataset of shape
    (nobservations, ndimensions)
    """
    def __init__(self, data, batch_size=100):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        new_data = [tf.constant(d) for d in self.data]
        data_iter_vb_new = tf.train.batch(new_data, self.batch_size,
                                          dynamic_pad=True)
        # data_iter_vb = [vb.eval() for vb in data_iter_vb_new]
        return iter(data_iter_vb_new)


class DatasetMiniBatchIterator(object):
    """Minibatch iterator over one dataset of shape
    (nobservations, ndimensions)
    """
    def __init__(self, data, batch_size, randomize=False):
        super(DatasetMiniBatchIterator, self).__init__()
        self.data = data  # tuple of datasets w/ same nobservations
        self.batch_size = batch_size
        self.randomize = randomize

    def __iter__(self):
        rows = range(len(self.data))
        if self.randomize:
            np.random.shuffle(rows)
        beg_indices = range(0, len(self.data) - self.batch_size + 1,
                            self.batch_size)
        end_indices = range(self.batch_size, len(self.data) + 1,
                            self.batch_size)
        for beg, end in zip(beg_indices, end_indices):
            curr_rows = rows[beg:end]
            yield self.data[curr_rows, :]


class MultiDatasetMiniBatchIterator(object):
    """Minibatch iterator over multiple datasets of shape
    (nobservations, ndimensions)
    """
    def __init__(self, data, batch_size, randomize=False):
        super(MultiDatasetMiniBatchIterator, self).__init__()
        self.data = data  # tuple of datasets w/ same nobservations
        self.batch_size = batch_size
        self.randomize = randomize

    def __iter__(self):
        rows = range(len(self.data[0]))
        if self.randomize:
            np.random.shuffle(rows)
        beg_indices = range(0, len(self.data[0]) - self.batch_size + 1,
                            self.batch_size)
        end_indices = range(self.batch_size, len(self.data[0]) + 1,
                            self.batch_size)
        for beg, end in zip(beg_indices, end_indices):
            curr_rows = rows[beg:end]
            yield tuple(dset[curr_rows, :] for dset in self.data)


class KLqp_profile(ed.KLqp):
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

        sess = ed.get_session()
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

        return {'t': t, 'loss': loss}

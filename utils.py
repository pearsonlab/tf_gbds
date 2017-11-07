import tensorflow as tf
import math
from os.path import expanduser
import h5py
from scipy.stats import norm
import numpy as np
import pandas as pd
from tensorflow.contrib.keras import layers as keras_layers
from tensorflow.contrib.keras import constraints, models
# import theano
# import theano.tensor as T
# import lasagne
# from lasagne.nonlinearities import rectify, linear
from matplotlib.colors import Normalize
# import sys
# sys.path.append(expanduser('~/code/gbds/code/'))
from tf_gbds.layers import PKBiasLayer, PKRowBiasLayer


class set_cbar_zero(Normalize):
    """
    set_cbar_zero(midpoint = float)       default: midpoint = 0.
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
    """
    Smoothing with gaussian convolution
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
    rtrial = trial.copy()
    for i in range(rtrial.shape[1]):
        rtrial[:, i] = gauss_convolve(rtrial[:, i], sigma,
                                      pad_method=pad_method)
    return rtrial


def get_session_names(file_loc, columns, values, comb=np.all):
    data_index = pd.read_csv(expanduser(file_loc), index_col=0)
    rows = comb([data_index[column] == value for column, value in zip(columns,
                                                                      values)],
                axis=0)
    return data_index[rows].index.values.tolist()


def load_pk_data(file_loc, session_names, train_split=0.85, get_spikes=False,
                 get_gaze=False, norm_x=True):
    """
    Load penaltykick data. norm_x flag converts range of x-dim from (0, 1) to
    (-1, 1)
    """
    datafile = h5py.File(expanduser(file_loc))

    sessions = map(lambda sess_name: datafile.get(sess_name),
                   session_names)

    modes = {0: [0, 0, 0, 0],  # normal
             1: [1, 0, 0, 0],  # saline and DLPFC
             2: [0, 1, 0, 0],  # saline and DMPFC
             3: [1, 0, 1, 0],  # muscimol and DLPFC
             4: [0, 1, 0, 1]}  # muscimol and DMPFC

    y_data = []
    y_data_modes = []
    y_val_data = []
    y_val_data_modes = []
    if get_spikes:
        y_data_spikes = []
        y_data_signals = []
        y_val_data_spikes = []
        y_val_data_signals = []
        signal_range = (0, 0)  # current signals
    if get_gaze:
        y_data_gaze = []
        y_val_data_gaze = []
    for sess in sessions:
        new_session = True  # assign new set of signals for each new session
        for key in sess.keys():
            info = sess.get(key).attrs
            if (info['Complete'] and info['GameMode'] == 1
                    and not info['ReplayOldBarData']):
                if get_spikes:  # if looking for trials with spike data
                    if not info['Spikes']:  # if trial does not have spike
                                            # data
                        continue  # skip
                if get_gaze:  # if looking for trials with gaze data
                    if not info['Gaze']:  # if trial does not have gaze data
                        continue  # skip
                if sess.get(key).value[1, :].max() <= 1.0:
                    if np.random.rand() <= train_split:
                        curr_data = y_data
                        curr_modes = y_data_modes
                        if get_spikes:
                            curr_spikes = y_data_spikes
                            curr_signals = y_data_signals
                        if get_gaze:
                            curr_gaze = y_data_gaze
                    else:
                        curr_data = y_val_data
                        curr_modes = y_val_data_modes
                        if get_spikes:
                            curr_spikes = y_val_data_spikes
                            curr_signals = y_val_data_signals
                        if get_gaze:
                            curr_gaze = y_val_data_gaze
                    raw_trial = sess.get(key).value[:3, :].T
                    if norm_x:
                        raw_trial[:, 1] = raw_trial[:, 1] * 2 - 1
                    trial = (smooth_trial(raw_trial)
                             .astype(np.float32))
                    curr_data.append(trial)

                    if get_spikes:
                        spike_entry = datafile.get('spikes' + sess.name +
                                                   '/' + key)
                        curr_spikes.append(spike_entry.value.T.astype('int32'))
                        if new_session:
                            # number of signals for session
                            ns = len(spike_entry.attrs['Signals'])
                            signal_range = (signal_range[1],
                                            signal_range[1] + ns)
                            new_session = False
                        if ns != len(spike_entry.attrs['Signals']):
                            # if signal added in middle of session
                            ns = len(spike_entry.attrs['Signals'])
                            signal_range = (signal_range[0],
                                            signal_range[0] + ns)
                        curr_signals.append(np.arange(*signal_range,
                                                      dtype='int32'))

                    if get_gaze:
                        gaze_entry = datafile.get('gaze' + sess.name + '/' +
                                                  key)
                        curr_gaze.append(gaze_entry.value.T
                                         .astype(np.float32))

                    if info['Ses_Type'] == 3 and info['Pre_Post'] == 2:
                        # post-saline injection
                        if info['Region'] == 1:  # DLPFC
                            curr_modes.append(modes[1])
                        elif info['Region'] == 2:  # DMPFC
                            curr_modes.append(modes[2])
                    elif info['Ses_Type'] == 2 and info['Pre_Post'] == 2:
                        # post-muscimol injection
                        if info['Region'] == 1:  # DLPFC
                            curr_modes.append(modes[3])
                        elif info['Region'] == 2:  # DMPFC
                            curr_modes.append(modes[4])
                    else:  # pre-injection or a recording trial
                        curr_modes.append(modes[0])

    y_data = np.array(y_data)
    y_data_modes = np.array(y_data_modes).astype('int32')
    y_val_data = np.array(y_val_data)
    y_val_data_modes = np.array(y_val_data_modes).astype('int32')
    rets = [y_data, y_data_modes, y_val_data, y_val_data_modes]
    if get_spikes:
        y_data_spikes = np.array(y_data_spikes)
        y_val_data_spikes = np.array(y_val_data_spikes)
        rets += [y_data_spikes, y_data_signals, y_val_data_spikes,
                 y_val_data_signals]
    if get_gaze:
        y_data_gaze = np.array(y_data_gaze)
        y_val_data_gaze = np.array(y_val_data_gaze)
        rets += [y_data_gaze, y_val_data_gaze]
    return rets


def get_max_velocities(y_data, y_val_data):
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
                              max(ball_y_vels)]), decimals=3)# + 0.001


def get_vel(data, max_vel=None):
    """
    Input a time series of positions and include velocities for each
    coordinate in each row
    """
    if isinstance(data, tf.Tensor):
        B = tf.shape(data)[0]
        dims = tf.shape(data)[-1]
        positions = data
    else:
        B = data.shape[0]
        dims = data.shape[-1]
        positions = data.copy()
    velocities = data[:, 1:] - data[:, :-1]
    if max_vel is not None:
        velocities /= max_vel
    velocities = tf.concat([tf.zeros((B, 1, dims), tf.float32), velocities],
                           1, name='velocities')
    states = tf.concat([positions, velocities], -1, name='position_vel')
    return states


def get_accel(data, max_vel=None):
    """
    Input a time series of positions and include velocities and acceleration
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


def get_network(name, input_dim, output_dim, hidden_dim, num_layers,
                PKLparams, batchnorm=False, is_shooter=False,
                row_sparse=False, add_pklayers=False, filt_size=None):
    """
    Returns a NN with the specified parameters.
    Also returns a list of PKBias layers
    """
    PKbias_layers = []
    NN = models.Sequential()
    with tf.name_scope('%s_input' % name):
        NN.add(keras_layers.InputLayer(input_shape=(None, input_dim),
                                       name='%s_Input' % name))

    with tf.name_scope('%s_batchnorm' % name):
        if batchnorm:
            NN.add(keras_layers.BatchNormalization(name='%s_BatchNorm' % name))
    with tf.name_scope('%s_filter' % name):
        if filt_size is not None:  # first layer convolution
            # rearrange dims for convolution
            # NN.add(keras_layers.Lambda(lambda x: tf.expand_dims(x, 0),
            #                            name='%s_ExpandDims' % name))
            # custom pad so that no timepoint gets input from future
            NN.add(keras_layers.ZeroPadding1D(padding=(filt_size - 1, 0),
                                              name='%s_ZeroPadding' % name))
            # Perform convolution (no pad, no filter flip (for interpretability))
            NN.add(keras_layers.Conv1D(
                filters=hidden_dim, kernel_size=filt_size, padding='valid',
                activation=tf.nn.relu,
                kernel_constraint=constraints.MaxNorm(5),
                bias_constraint=constraints.MaxNorm(5),
                name='%s_Conv1D' % name))
            # rearrange dims for dense layers
            # NN.add(keras_layers.Lambda(lambda x: tf.squeeze(x, [0]),
            #                            name='%s_Squeeze' % name))
    with tf.name_scope('%s_layers' % name):
        for i in range(num_layers):
            with tf.name_scope('%s_PK_Bias' % name):
                if is_shooter and add_pklayers:
                    if row_sparse:
                        PK_bias = PKRowBiasLayer(
                            NN, PKLparams,
                            name='%s_PKRowBias_%s' % (name, i+1))
                    else:
                        PK_bias = PKBiasLayer(
                            NN, PKLparams,
                            name='%s_PKBias_%s' % (name, i+1))
                    PKbias_layers.append(PK_bias)
                    NN.add(PK_bias)

            if i == num_layers - 1:
                NN.add(keras_layers.Dense(
                    output_dim, name='%s_Dense_%s' % (name, i+1),
                    kernel_initializer=tf.random_normal_initializer(
                        stddev=0.1),
                    kernel_constraint=constraints.MaxNorm(5),
                    bias_constraint=constraints.MaxNorm(5)))
                NN.add(keras_layers.Activation(
                    activation='linear',
                    name='%s_Activation_%s' % (name, i+1)))
            else:
                NN.add(keras_layers.Dense(
                    hidden_dim, name='%s_Dense_%s' % (name, i+1),
                    kernel_initializer=tf.orthogonal_initializer(),
                    kernel_constraint=constraints.MaxNorm(5),
                    bias_constraint=constraints.MaxNorm(5)))
                NN.add(keras_layers.Activation(
                    activation='relu',
                    name='%s_Activation_%s' % (name, i+1)))
    return NN, PKbias_layers


def get_rec_params_GBDS(obs_dim, lag, num_layers, hidden_dim, penalty_Q,
                        PKLparams, name):

    with tf.name_scope('rec_mu_%s' % name):
        mu_net, PKbias_layers_mu = get_network('rec_mu_%s' % name,
                                               obs_dim * (lag + 1),
                                               obs_dim, hidden_dim,
                                               num_layers, PKLparams,
                                               batchnorm=False)

    with tf.name_scope('rec_lambda_%s' % name):
        lambda_net, PKbias_layers_lambda = get_network('rec_lambda_%s' % name,
                                                       obs_dim * (lag + 1),
                                                       obs_dim**2,
                                                       hidden_dim,
                                                       num_layers, PKLparams,
                                                       batchnorm=False)

    with tf.name_scope('rec_lambdaX_%s' % name):
        lambdaX_net, PKbias_layers_lambdaX = get_network('rec_lambdaX_%s' % name,
                                                         obs_dim * (lag + 1),
                                                         obs_dim**2,
                                                         hidden_dim,
                                                         num_layers, PKLparams,
                                                         batchnorm=False)

    rec_params = dict(A=.9 * np.eye(obs_dim),
                      QinvChol=np.eye(obs_dim),
                      Q0invChol=np.eye(obs_dim),
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


# def get_gen_params_GBDS(obs_dim_agent, obs_dim, add_accel,
#                         yCols_agent, num_layers_rec, hidden_dim, PKLparams,
#                         vel, penalty_eps, penalty_sigma,
#                         boundaries_g, penalty_g, name):

#     with tf.name_scope('get_states_%s' % name):
#         if add_accel:
#             get_states = get_accel
#         else:
#             get_states = get_vel

#     with tf.name_scope('gen_mu_%s' % name):
#         NN_postJ_mu, _ = get_network('gen_mu_%s' % name,
#                                      obs_dim_agent * 2,
#                                      obs_dim_agent * 2,
#                                      hidden_dim,
#                                      num_layers_rec,
#                                      PKLparams)

#     with tf.name_scope('gen_sigma_%s' % name):
#         NN_postJ_sigma, _ = get_network('gen_sigma_%s' % name,
#                                         obs_dim_agent * 2,
#                                         (obs_dim_agent * 2)**2,
#                                         hidden_dim,
#                                         num_layers_rec,
#                                         PKLparams)

#     gen_params = dict(vel=vel[yCols_agent],
#                       yCols=yCols_agent,  # which columns belong to the agent
#                       pen_eps=penalty_eps,
#                       pen_sigma=penalty_sigma,
#                       bounds_g=boundaries_g,
#                       pen_g=penalty_g,
#                       NN_postJ_mu=NN_postJ_mu,
#                       NN_postJ_sigma=NN_postJ_sigma,
#                       get_states=get_states)

#     return gen_params


def get_gen_params_GBDS_GMM(obs_dim_agent, obs_dim, add_accel,
                            yCols_agent, nlayers_gen, hidden_dim_gen,
                            K, C, PKLparams, vel,
                            penalty_eps, penalty_sigma, penalty_A,
                            boundaries_g, penalty_g,
                            clip, clip_range, clip_tol, eta, name):

    with tf.name_scope('get_states_%s' % name):
        if add_accel:
            get_states = get_accel
            state_dim = obs_dim * 3
        else:
            get_states = get_vel
            state_dim = obs_dim * 2

    with tf.name_scope('gen_gmm_%s' % name):
        GMM_net_1, _ = get_network('gen_gmm_%s' % name, state_dim,
                                   (obs_dim_agent * K * 2),
                                   hidden_dim_gen, nlayers_gen, PKLparams)
        GMM_net_2, _ = get_network('gen_gmm_%s' % name, state_dim,
                                   C * K ** 2, hidden_dim_gen,
                                   nlayers_gen, PKLparams)

    gen_params = dict(all_vel=vel,
                      vel=vel[yCols_agent],
                      yCols=yCols_agent,  # which columns belong to the agent
                      pen_eps=penalty_eps,
                      pen_sigma=penalty_sigma,
                      pen_A=penalty_A,
                      bounds_g=boundaries_g,
                      pen_g=penalty_g,
                      clip=clip,
                      clip_range=clip_range,
                      clip_tol=clip_tol,
                      eta=eta,
                      get_states=get_states,
                      GMM_net_1=GMM_net_1,
                      GMM_net_2=GMM_net_2,
                      K=K, C=C)

    return gen_params


def initialize_PID_params(player, Dim):
    PID_params = {}
    PID_params['unc_Kp'] = tf.Variable(initial_value=np.zeros((Dim, 1)),
                                       name='unc_Kp_' + player,
                                       dtype=tf.float32)
    PID_params['unc_Ki'] = tf.Variable(initial_value=np.zeros((Dim, 1)),
                                       name='unc_Ki_' + player,
                                       dtype=tf.float32)
    PID_params['unc_Kd'] = tf.Variable(initial_value=np.zeros((Dim, 1)),
                                       name='unc_Kd_' + player,
                                       dtype=tf.float32)
    PID_params['unc_eps'] = tf.Variable(
        initial_value=-11.513 * np.ones((1, Dim)), name='unc_eps_' + player,
        dtype=tf.float32)

    return PID_params


def initialize_Dyn_params(player, RecognitionParams):
    Dyn_params = {}
    Dyn_params['A'] = tf.Variable(RecognitionParams['A'], name='A_' + player,
                                  dtype=tf.float32)
    Dyn_params['QinvChol'] = tf.Variable(RecognitionParams['QinvChol'],
                                         name='QinvChol_' + player,
                                         dtype=tf.float32)
    Dyn_params['Q0invChol'] = tf.Variable(RecognitionParams['Q0invChol'],
                                          name='Q0invChol_' + player,
                                          dtype=tf.float32)

    return Dyn_params


# def compile_functions_vb_training(model, learning_rate, add_pklayers=False,
#                                   joint_spikes=False, cap_noise=False,
#                                    mode='FAST_RUN'):
#     # Define a bare-bones theano training function
#     batch_y = tf.placeholder(tf.float32, shape=(None, None), name='batch_y')
#     givens = {model.Y: batch_y}
#     inputs = [theano.In(batch_y)]
#     if add_pklayers:
#         batch_mode = T.ivector('batch_mode')
#         givens[model.mode] = batch_mode
#         inputs += [theano.In(batch_mode)]
#     if joint_spikes and model.isTrainingSpikeModel:
#         batch_spikes = T.imatrix('batch_spikes')
#         batch_signals = T.ivector('batch_signals')
#         givens[model.spikes] = batch_spikes
#         givens[model.signals] = batch_signals
#         inputs += [theano.In(batch_spikes), theano.In(batch_signals)]

#     # use lasagne to get adam updates, and constrain norms
#     # control model functions
#     model.set_training_mode('CTRL')
#     ctrl_updates = lasagne.updates.adam(-model.cost(), model.getParams(),
#     learning_rate=learning_rate)
#     for param in model.getParams():
#         if param.name == 'W' or param.name == 'b' or param.name == 'G':
#         # only on NN parameters
#             ctrl_updates[param] = lasagne.updates
#                                    .norm_constraint(ctrl_updates[param],
#                                                     max_norm=5,
#                                                      norm_axes=[0])
#         if cap_noise and (param.name == 'QinvChol' or
#                            param.name == 'Q0invChol'):
#             ctrl_updates[param] = lasagne.updates
#                                    .norm_constraint(ctrl_updates[param],
#                                                     max_norm=30,
#                                                     norm_axes=[0])

#     # Finally, compile the function that will actually take gradient steps.
#     ctrl_train_fn = theano.function(
#         outputs=model.cost(),
#         inputs=inputs,
#         updates=ctrl_updates,
#         givens=givens,
#         mode=mode)
#     ctrl_test_fn = theano.function(
#         outputs=model.cost(),
#         inputs=inputs,
#         givens=givens)

#     return ctrl_train_fn, ctrl_test_fn


# def compile_functions_cgan_training(model, learning_rate, mode='FAST_RUN'):
#     batch_J, batch_s = T.matrices('batch_J', 'batch_s')

#     # GAN generator training function
#     model.set_training_mode('CGAN_G')
#     gan_g_updates = lasagne.updates.adam(-model.cost(), model.getParams(),
#                                          learning_rate=learning_rate)
#     for param in model.getParams():
#         if param.name == 'W' or param.name == 'b' or param.name == 'G':
#          # only on NN parameters
#             gan_g_updates[param] = lasagne.updates
#                                     .norm_constraint(gan_g_updates[param],
#                                                      max_norm=5,
#                                                      norm_axes=[0])

#     # Finally, compile the function that will actually take gradient steps.
#     gan_g_train_fn = theano.function(
#         outputs=model.cost(),
#         inputs=[theano.In(batch_s)],
#         updates=gan_g_updates,
#         givens={model.s: batch_s},
#         mode=mode)

#     # GAN discriminator training function
#     model.set_training_mode('CGAN_D')
#     gan_d_updates = lasagne.updates.adam(-model.cost(), model.getParams(),
#                                          learning_rate=learning_rate)

#     # Finally, compile the function that will actually take gradient steps.
#     gan_d_train_fn = theano.function(
#         outputs=model.cost(),
#         inputs=[theano.In(batch_J), theano.In(batch_s)],
#         updates=gan_d_updates,
#         givens={model.J: batch_J, model.s: batch_s},
#         mode=mode)

#     return gan_g_train_fn, gan_d_train_fn


# def compile_functions_gan_training(model, learning_rate, mode='FAST_RUN'):
#     batch_g0 = T.matrix('batch_g0')

#     # GAN generator training function
#     model.set_training_mode('GAN_G')
#     gan_g_updates = lasagne.updates.adam(-model.cost(), model.getParams(),
#                                          learning_rate=learning_rate)
#     for param in model.getParams():
#         if param.name == 'W' or param.name == 'b' or param.name == 'G':
#               # only on NN parameters
#             gan_g_updates[param] = lasagne.updates
#              .norm_constraint(gan_g_updates[param],
#                               max_norm=5, norm_axes=[0])

#     # Finally, compile the function that will actually take gradient steps.
#     gan_g_train_fn = theano.function(
#         outputs=model.cost(),
#         inputs=[theano.In(batch_g0)],
#         updates=gan_g_updates,
#         givens={model.g0: batch_g0},
#         mode=mode)

#     # GAN discriminator training function
#     model.set_training_mode('GAN_D')
#     gan_d_updates = lasagne.updates.adam(-model.cost(), model.getParams(),
#                                          learning_rate=learning_rate)

#     # Finally, compile the function that will actually take gradient steps.
#     gan_d_train_fn = theano.function(
#         outputs=model.cost(),
#         inputs=[theano.In(batch_g0)],
#         updates=gan_d_updates,
#         givens={model.g0: batch_g0},
#         mode=mode)

#     return gan_g_train_fn, gan_d_train_fn

def gen_data(n_trial, n_obs, sigma=np.log1p(np.exp(-5 * np.ones((1, 2)))),
             eps=1e-5, Kp=1, Ki=0, Kd=0,
             vel=1e-2 * np.ones((3))):
    p = []
    g_b = []

    for s in range(n_trial):
        p_b = np.zeros((n_obs, 2), np.float32)
        p_g = np.zeros((n_obs, 1), np.float32)
        g = np.zeros(p_b.shape, np.float32)
        prev_error_b = 0
        prev_error_g = 0
        int_error_b = 0
        int_error_g = 0

        init = np.pi * (np.random.rand() * 2 - 1)
        g_mu_y = 0.75 * np.sin(1. * (np.linspace(0, 2 * np.pi, n_obs) - init))
        g_mu = np.hstack([np.ones((n_obs, 1)), g_mu_y.reshape(n_obs, 1)])
        g_lambda = np.array([16, 16], np.float32)
        g[0] = [1, 0.75 * (np.random.rand() * 2 - 1)]

        for t in range(n_obs - 1):
            g[t + 1] = (g[t] + g_lambda * g_mu[t + 1]) / (1 + g_lambda)
            var = sigma ** 2 / (1 + g_lambda)
            g[t + 1] += (np.random.randn(1, 2) * np.sqrt(var)).reshape(2,)

            error_b = g[t + 1] - p_b[t]
            int_error_b += error_b
            der_error_b = error_b - prev_error_b
            u_b = (Kp * error_b + Ki * int_error_b + Kd * der_error_b +
                   eps * np.random.randn(2,))
            prev_error_b = error_b
            p_b[t + 1] = p_b[t] + vel[1:] * np.clip(u_b, -1, 1)

            error_g = p_b[t + 1, 1] - p_g[t]
            int_error_g += error_g
            der_error_g = error_g - prev_error_g
            u_g = (Kp * error_g + Ki * int_error_g + Kd * der_error_g +
                   eps * np.random.randn())
            prev_error_g = error_g
            p_g[t + 1] = p_g[t] + vel[0] * np.clip(u_g, -1, 1)

        p.append(np.hstack([p_g, p_b]))
        g_b.append(g)

    return p, g_b


def generator(arrays, batch_size):
    size = len(arrays)
    n_batch = math.ceil(size / batch_size)
    start = 0
    while True:
        batches = []
        for _ in range(n_batch):
            stop = start + batch_size
            diff = stop - size
            if diff <= 0:
                batch = np.array(arrays[start:stop])
                start = stop
            else:
                batch = np.array(arrays[start:])
            batches.append(batch)
        yield batches


class DatasetTrialIndexIterator(object):
    """ Basic trial iterator """
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
    """
    Trial iterator over multiple datasets of shape
    (ntrials, trial_len, trial_dimensions)
    """
    def __init__(self, data, randomize=False, batch_size=1):
        self.data = data
        self.randomize = randomize

    def __iter__(self):
        n_batches = len(self.data[0])
        if self.randomize:
            indices = range(n_batches)
            np.random.shuffle(indices)
            for i in indices:
                yield tuple(dset[i] for dset in self.data)
        else:
            for i in range(n_batches):
                yield tuple(dset[i] for dset in self.data)
                

class MultiDataSetTrialIndexTF(object):
    def __init__(self, data, batch_size=100):
        self.data = data
        self.batch_size = batch_size
    def __iter__(self):
        new_data = [tf.constant(d) for d in self.data]
        data_iter_vb_new = tf.train.batch(new_data, self.batch_size,
                                          dynamic_pad=True)
        data_iter_vb = [vb.eval() for vb in data_iter_vb_new]
        return iter(data_iter_vb)


class DatasetMiniBatchIterator(object):
    """
    Minibatch iterator over one dataset of shape
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
    """
    Minibatch iterator over multiple datasets of shape
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
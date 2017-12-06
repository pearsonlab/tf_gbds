import tensorflow as tf
import math
from os.path import expanduser
import h5py
from scipy.stats import norm
import numpy as np
import pandas as pd
from tensorflow.contrib.keras import layers as keras_layers
from tensorflow.contrib.keras import constraints, models
from matplotlib.colors import Normalize
from layers import PKBiasLayer, PKRowBiasLayer
import edward as ed
import six

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

# def load_pkhuman_data_pickle(hps, train_split=0.85):
#     """
#     Load penaltykick data from human fMRI Huettel task. norm_x flag converts 
#     range of x-dim from (0, 1) to (-1, 1)
#     """
#     #datafile = h5py.File(expanduser(file_loc))
#     file_loc = hps.data_loc
#     cutBegin = hps.cut_Begin
#     norm_x = True
#     np.random.seed(hps.seed)  # set seed for consistent train/val split
#     with open(file_loc, 'rb') as pickle_file:
#         datafile = pickle.load(pickle_file, encoding='latin1') #encoding param to deal with python 2/3 pickle issues

#     #sessions = map(lambda sess_name: datafile.get(sess_name),
#     #               session_names)
#     nSubs = datafile.subj.nunique() #how many subjects are in this dataset?
#     s = np.arange(nSubs)
#     hotcode = pd.get_dummies(s)
#     modes = {} #dummies representing each subject
#     for i in s:
#         modes[int(datafile.subj.unique()[i])] = hotcode.loc[[i]]

#     y_data = []
#     y_data_modes = []
#     y_val_data = []
#     y_val_data_modes = []
#     y_data_res = []
#     y_val_data_res = []
#     y_data_JS = []
#     y_val_data_JS = []    
#     y_data_opp = []
#     y_val_data_opp = []

#     #y_data = [content[content.super_index = i] for i in set(content.super_index)]
#     for trial in set(datafile.super_index): #for each trial
#         info = datafile[datafile.super_index == trial] #all of the data corresponding to this trial
#         if np.random.rand() <= train_split:
#             curr_data = y_data
#             curr_modes = y_data_modes        
#             curr_res = y_data_res
#             curr_JS = y_data_JS  
#             curr_opp = y_data_opp              
#         else:
#             curr_data = y_val_data
#             curr_modes = y_val_data_modes
#             curr_res = y_val_data_res
#             curr_JS = y_val_data_JS
#             curr_opp = y_val_data_opp
#         #raw_trial = sess.get(trial).value[:3, :].T
#         raw_trial = info[['goalie','time','ball_alone']].values #grab the goalie y position, ball x position (same as time), and ball y position

#         #JS_raw = info[['barY_JS', 'ballY_JS']].values
#         is_win = (info['result'].iloc[0] == 'W')
#         is_human = (info['opponent'].iloc[0] == 'human')
#         curr_opp.append(is_human)
#         curr_res.append(is_win)

#         if is_human:
#             JS_raw = info[['barY_JS', 'ballY_JS']].values
#             #print JS_raw.shape
#         if not is_human: #if the goalie is a computer, we need to calculate the "joystick inputs"
#             barY_JS = np.hstack((0,np.diff(info['goalie'])/(info[['maxMove']].values[1:]).flatten()))
#             ballY_JS = info[['ballY_JS']].values.flatten()
#             tmp = pd.DataFrame({'barY_JS':barY_JS, 'ballY_JS':ballY_JS})
#             JS_raw = tmp[['barY_JS','ballY_JS']].values
#             #print barY_JS.shape, ballY_JS.shape
#             #JS_raw = info[['barY_JS', 'ballY_JS']].values

#         if norm_x:
#             raw_trial[:, 0] = (raw_trial[:, 0] / 768.) * 2 - 1
#             if is_win:
#                 end_x = 920.0
#             else:
#                 end_x = 896.0
#             raw_trial[:, 1] = np.linspace(128./1024., end_x/1024., len(raw_trial)) * 2 - 1
#             raw_trial[:, 2] = (raw_trial[:, 2] / 768.) * 2 - 1 #normalize the y-dimension as well
#         raw_trial = raw_trial[cutBegin:,:] #cut x timepoints, where x is the cutBegin arg. This handles the "zero activity" at the beginning
#         JS_raw = JS_raw[cutBegin:,:]
#         #JS = raw_trial[:,3:] #joystick data for goalie and ball
#         #trial = smooth_trial(raw_trial).astype(theano.config.floatX)
#         curr_data.append(np.array(raw_trial).astype('float32'))
#         curr_JS.append(np.array(JS_raw).astype('float32'))
#         curr_modes.append(np.array(modes[int(info.subj.iloc[0])]))
#         #curr_JS.append(np.array(raw_trial[:,3:]).astype('float32'))

#     #y_data = np.array(y_data)
#     #y_data_modes = np.array(y_data_modes).astype(theano.config.floatX)
#     y_data_modes = np.array(y_data_modes).astype('float32')
#     y_val_data = np.array(y_val_data)
#     #y_val_data_modes = np.array(y_val_data_modes).astype(theano.config.floatX)
#     y_val_data_modes = np.array(y_val_data_modes).astype('float32')
#     y_data_JS = np.array(y_data_JS)
#     y_val_data_JS = np.array(y_val_data_JS)
#     rets = [y_data, y_data_modes, y_data_res, y_data_JS, y_data_opp, y_val_data, y_val_data_modes, y_val_data_res, y_val_data_JS, y_val_data_opp]
#     return rets

# def load_PKhuman_hdf5(hps):
# 	'''
# 	Reads in the human data from PenaltyKik.02 from an hdf5 file.
# 	This is the standard format we will expect the data to be in.
# 	'''
# 	data = h5py.File('humanPK.h5','r')
# 	trajectories = np.array(data.get('trajectories'))
# 	states = np.array(data.get('states'))
# 	conditions = np.array(data.get('conditions'))
# 	outcomes = np.array(data.get('outcomes'))

# 	trajectories_train, trajectories_test = train_test_split(trajectories, test_size=1-hps.train_ratio, random_state=1)
# 	states_train, states_test = train_test_split(states, test_size=1-hps.train_ratio, random_state=1)
# 	conditions_train, conditions_test = train_test_split(conditions, test_size=1-hps.train_ratio, random_state=1)
# 	outcomes_train, outcomes_test = train_test_split(outcomes, test_size=1-hps.train_ratio, random_state=1)

# 	return [trajectories_train, trajectories_test, states_train, states_test, conditions_train, conditions_test, outcomes_train, outcomes_test]

# def get_session_names(file_loc, columns, values, comb=np.all):
#     """Get session names from real data
#     """
#     data_index = pd.read_csv(expanduser(file_loc), index_col=0)
#     rows = comb([data_index[column] == value for column, value in zip(columns,
#                                                                       values)],
#                 axis=0)
#     return data_index[rows].index.values.tolist()

def load_data(hps):
    """ Generate synthetic data set or load real data from local directory
    """
    train_data = []
    val_data = []
    if hps.synthetic_data:
        data, goals = gen_data(
            n_trial=2000, n_obs=100, Kp=0.6, Ki=0.3, Kd=0.1)
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
        if "conditions" in datafile and hps.add_conds:
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


def get_vel(data, max_vel=None):
    """Input a time series of positions and include velocities for each
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
        #max_vel = [max_vel[0], max_vel[2]] #lazily handle the clipping of only two dims for human data
        #import pdb; pdb.set_trace()
        velocities /= max_vel
    velocities = tf.concat([tf.zeros((B, 1, dims), tf.float32), velocities],
                           1, name='velocities')
    states = tf.concat([positions, velocities], -1, name='position_vel')
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


def get_network(name, input_dim, output_dim, hidden_dim, num_layers,
                PKLparams, batchnorm=False, is_shooter=False,
                row_sparse=False, add_pklayers=False, filt_size=None):
    """Returns a NN with the specified parameters.
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
            # Perform convolution (no pad, no filter flip
            # (for interpretability))
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


def get_rec_params_GBDS(obs_dim, extra_dim, lag, num_layers, hidden_dim, penalty_Q,
                        PKLparams, name):
    """Return a dictionary of timeseries-specific parameters for recognition
       model
    """

    with tf.name_scope('rec_mu_%s' % name):
        mu_net, PKbias_layers_mu = get_network('rec_mu_%s' % name,
                                               (obs_dim * (lag + 1) + extra_dim),
                                               obs_dim, hidden_dim,
                                               num_layers, PKLparams,
                                               batchnorm=False)

    with tf.name_scope('rec_lambda_%s' % name):
        lambda_net, PKbias_layers_lambda = get_network('rec_lambda_%s' % name,
                                                       (obs_dim * (lag + 1) + extra_dim),
                                                       obs_dim**2,
                                                       hidden_dim,
                                                       num_layers, PKLparams,
                                                       batchnorm=False)

    with tf.name_scope('rec_lambdaX_%s' % name):
        lambdaX_net, PKbias_layers_lambdaX = get_network('rec_lambdaX_%s'
                                                         % name,
                                                         (obs_dim * (lag + 1)+extra_dim),
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


def get_gen_params_GBDS_GMM(obs_dim_agent, obs_dim, extra_dim, add_accel,
                            yCols_agent, nlayers_gen, hidden_dim_gen,
                            K, PKLparams, vel,
                            penalty_eps, penalty_sigma,
                            boundaries_g, penalty_g,
                            clip, clip_range, clip_tol, eta, name):
    """Return a dictionary of timeseries-specific parameters for generative
       model
    """

    with tf.name_scope('get_states_%s' % name):
        if add_accel:
            get_states = get_accel
            state_dim = obs_dim * 3
        else:
            get_states = get_vel
            state_dim = obs_dim * 2

    with tf.name_scope('gen_gmm_%s' % name):
        GMM_net, _ = get_network('gen_gmm_%s' % name, (state_dim + extra_dim),
                                 (obs_dim_agent * K * 2 + K),
                                 hidden_dim_gen, nlayers_gen, PKLparams)
    with tf.name_scope('initial_distribution_%s' % name):
        g0_params = init_GMM_params(name, obs_dim_agent, K)

    gen_params = dict(all_vel=vel,
                      vel=vel[yCols_agent],
                      yCols=yCols_agent,  # which columns belong to the agent
                      pen_eps=penalty_eps,
                      pen_sigma=penalty_sigma,
                      bounds_g=boundaries_g,
                      pen_g=penalty_g,
                      clip=clip,
                      clip_range=clip_range,
                      clip_tol=clip_tol,
                      eta=eta,
                      get_states=get_states,
                      g0_params=g0_params,
                      GMM_net=GMM_net,
                      GMM_k=K)

    return gen_params


def init_PID_params(player, Dim):
    """Return a dictionary of PID controller parameters
    """
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
        initial_value=-10 * np.ones((1, Dim)), name='unc_eps_' + player,
        dtype=tf.float32)

    return PID_params


def init_Dyn_params(player, RecognitionParams):
    """Return a dictionary of dynamical parameters
    """
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
    
def init_GMM_params(player, Dim, K):
    GMM_params = {}
    GMM_params['K'] = K
    GMM_params['mu'] = tf.Variable(tf.random_normal([K, Dim]),
                                   dtype=tf.float32,
                                   name='g0_mu_%s' % player)
    GMM_params['unc_lambda'] = tf.Variable(tf.random_normal([K, Dim]),
                                           dtype=tf.float32,
                                           name='g0_unc_lambda_%s' % player)
    GMM_params['unc_w'] = tf.Variable(tf.ones([K]), dtype=tf.float32,
                                      name='g0_unc_w_%s' % player)

    return GMM_params

def batch_generator(arrays, batch_size, randomize=True):
    """Minibatch generator over one dataset of shape
    (#observations, #dimensions)
    """
    n_trials = len(arrays)
    n_batch = math.ceil(n_trials / batch_size)
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
            else:
                batch = np.array(arrays[start:])
            batches.append(batch)
        yield batches


def batch_generator_pad(arrays, batch_size, extra_conds=None, ctrl_obs=None,
                        randomize=True):
    n_trials = len(arrays)
    n_batch = math.ceil(n_trials / batch_size)
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
            else:
                batch = arrays[start:]
                if extra_conds is not None:
                    cond = np.array(extra_conds[start:])
                if ctrl_obs is not None:
                    ctrl = np.array(ctrl_obs[start:])
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
    elif mode =='zero':
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
        self.options=options
        self.run_metadata=run_metadata

    def update(self, feed_dict=None):
        if feed_dict is None:
          feed_dict = {}

        for key, value in six.iteritems(self.data):
          if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
            feed_dict[key] = value

        sess = ed.get_session()
        _, t, loss = sess.run([self.train, self.increment_t, self.loss],
                              options = self.options,
                              run_metadata = self.run_metadata,
                              feed_dict=feed_dict)

        if self.debug:
          sess.run(self.op_check, feed_dict)

        if self.logging and self.n_print != 0:
          if t == 1 or t % self.n_print == 0:
            summary = sess.run(self.summarize, feed_dict)
            self.train_writer.add_summary(summary, t)

        return {'t': t, 'loss': loss}


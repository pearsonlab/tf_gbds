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

<<<<<<< 007323809df8ff732133a86a38f3a3d9fb2d6cbd
=======
def load_pkhuman_data_pickle(hps, train_split=0.85):
    """
    Load penaltykick data from human fMRI Huettel task. norm_x flag converts 
    range of x-dim from (0, 1) to (-1, 1)
    """
    #datafile = h5py.File(expanduser(file_loc))
    file_loc = hps.data_loc
    cutBegin = hps.cut_Begin
    norm_x = True
    np.random.seed(hps.seed)  # set seed for consistent train/val split
    with open(file_loc, 'rb') as pickle_file:
        datafile = pickle.load(pickle_file, encoding='latin1') #encoding param to deal with python 2/3 pickle issues

    #sessions = map(lambda sess_name: datafile.get(sess_name),
    #               session_names)
    nSubs = datafile.subj.nunique() #how many subjects are in this dataset?
    s = np.arange(nSubs)
    hotcode = pd.get_dummies(s)
    modes = {} #dummies representing each subject
    for i in s:
        modes[int(datafile.subj.unique()[i])] = hotcode.loc[[i]]

    y_data = []
    y_data_modes = []
    y_val_data = []
    y_val_data_modes = []
    y_data_res = []
    y_val_data_res = []
    y_data_JS = []
    y_val_data_JS = []    
    y_data_opp = []
    y_val_data_opp = []

    #y_data = [content[content.super_index = i] for i in set(content.super_index)]
    for trial in set(datafile.super_index): #for each trial
        info = datafile[datafile.super_index == trial] #all of the data corresponding to this trial
        if np.random.rand() <= train_split:
            curr_data = y_data
            curr_modes = y_data_modes        
            curr_res = y_data_res
            curr_JS = y_data_JS  
            curr_opp = y_data_opp              
        else:
            curr_data = y_val_data
            curr_modes = y_val_data_modes
            curr_res = y_val_data_res
            curr_JS = y_val_data_JS
            curr_opp = y_val_data_opp
        #raw_trial = sess.get(trial).value[:3, :].T
        raw_trial = info[['goalie','time','ball_alone']].values #grab the goalie y position, ball x position (same as time), and ball y position

        #JS_raw = info[['barY_JS', 'ballY_JS']].values
        is_win = (info['result'].iloc[0] == 'W')
        is_human = (info['opponent'].iloc[0] == 'human')
        curr_opp.append(is_human)
        curr_res.append(is_win)

        if is_human:
            JS_raw = info[['barY_JS', 'ballY_JS']].values
            #print JS_raw.shape
        if not is_human: #if the goalie is a computer, we need to calculate the "joystick inputs"
            barY_JS = np.hstack((0,np.diff(info['goalie'])/(info[['maxMove']].values[1:]).flatten()))
            ballY_JS = info[['ballY_JS']].values.flatten()
            tmp = pd.DataFrame({'barY_JS':barY_JS, 'ballY_JS':ballY_JS})
            JS_raw = tmp[['barY_JS','ballY_JS']].values
            #print barY_JS.shape, ballY_JS.shape
            #JS_raw = info[['barY_JS', 'ballY_JS']].values

        if norm_x:
            raw_trial[:, 0] = (raw_trial[:, 0] / 768.) * 2 - 1
            if is_win:
                end_x = 920.0
            else:
                end_x = 896.0
            raw_trial[:, 1] = np.linspace(128./1024., end_x/1024., len(raw_trial)) * 2 - 1
            raw_trial[:, 2] = (raw_trial[:, 2] / 768.) * 2 - 1 #normalize the y-dimension as well
        raw_trial = raw_trial[cutBegin:,:] #cut x timepoints, where x is the cutBegin arg. This handles the "zero activity" at the beginning
        JS_raw = JS_raw[cutBegin:,:]
        #JS = raw_trial[:,3:] #joystick data for goalie and ball
        #trial = smooth_trial(raw_trial).astype(theano.config.floatX)
        curr_data.append(np.array(raw_trial).astype('float32'))
        curr_JS.append(np.array(JS_raw).astype('float32'))
        curr_modes.append(np.array(modes[int(info.subj.iloc[0])]))
        #curr_JS.append(np.array(raw_trial[:,3:]).astype('float32'))

    #y_data = np.array(y_data)
    #y_data_modes = np.array(y_data_modes).astype(theano.config.floatX)
    y_data_modes = np.array(y_data_modes).astype('float32')
    y_val_data = np.array(y_val_data)
    #y_val_data_modes = np.array(y_val_data_modes).astype(theano.config.floatX)
    y_val_data_modes = np.array(y_val_data_modes).astype('float32')
    y_data_JS = np.array(y_data_JS)
    y_val_data_JS = np.array(y_val_data_JS)
    rets = [y_data, y_data_modes, y_data_res, y_data_JS, y_data_opp, y_val_data, y_val_data_modes, y_val_data_res, y_val_data_JS, y_val_data_opp]
    return rets

def load_PKhuman(hps):
	'''
	Reads in the human data from PenaltyKik.02 from an hdf5 file.
	This is the standard format we will expect the data to be in.
	'''
	data = h5py.File('humanPK.h5','r')
	trajectories = np.array(data.get('trajectories'))
	states = np.array(data.get('states'))
	conditions = np.array(data.get('conditions'))
	outcomes = np.array(data.get('outcomes'))

	trajectories_train, trajectories_test = train_test_split(trajectories, test_size=1-hps.train_ratio, random_state=1)
	states_train, states_test = train_test_split(states, test_size=1-hps.train_ratio, random_state=1)
	conditions_train, conditions_test = train_test_split(conditions, test_size=1-hps.train_ratio, random_state=1)
	outcomes_train, outcomes_test = train_test_split(outcomes, test_size=1-hps.train_ratio, random_state=1)

	return [trajectories_train, trajectories_test, states_train, states_test, conditions_train, conditions_test, outcomes_train, outcomes_test]

>>>>>>> changes
def get_session_names(file_loc, columns, values, comb=np.all):
    """Get session names from real data
    """
    data_index = pd.read_csv(expanduser(file_loc), index_col=0)
    rows = comb([data_index[column] == value for column, value in zip(columns,
                                                                      values)],
                axis=0)
    return data_index[rows].index.values.tolist()

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


def get_rec_params_GBDS(obs_dim, lag, num_layers, hidden_dim, penalty_Q,
                        PKLparams, name):
    """Return a dictionary of timeseries-specific parameters for recognition
       model
    """

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
        lambdaX_net, PKbias_layers_lambdaX = get_network('rec_lambdaX_%s'
                                                         % name,
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


def get_gen_params_GBDS_GMM(obs_dim_agent, obs_dim, add_accel,
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
        GMM_net, _ = get_network('gen_gmm_%s' % name, state_dim,
                                 (obs_dim_agent * K * 2 + K),
                                 hidden_dim_gen, nlayers_gen, PKLparams)

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


def gen_data(n_trial, n_obs, sigma=np.log1p(np.exp(-5 * np.ones((1, 2)))),
             eps=np.log1p(np.exp(-10.)), Kp=1, Ki=0, Kd=0,
             vel=1e-2 * np.ones((3))):
    """Generate fake data to test the accuracy of the model
    """
    p = []
    g = []

    for _ in range(n_trial):
        p_b = np.zeros((n_obs, 2), np.float32)
        p_g = np.zeros((n_obs, 1), np.float32)
        g_b = np.zeros((n_obs, 2), np.float32)
        prev_error_b = 0
        prev_error_g = 0
        int_error_b = 0
        int_error_g = 0

        init_b_x = np.pi * (np.random.rand() * 2 - 1)
        g_b_x_mu = 0.25 * np.sin(2. * (np.linspace(0, 2 * np.pi, n_obs) -
                                       init_b_x))
        init_b_y = np.pi * (np.random.rand() * 2 - 1)
        g_b_y_mu = 0.25 * np.sin(2. * (np.linspace(0, 2 * np.pi, n_obs) -
                                       init_b_y))
        g_b_mu = np.hstack([g_b_x_mu.reshape(n_obs, 1), g_b_y_mu.reshape(n_obs,
                                                                         1)])
        g_b_lambda = np.array([16, 16], np.float32)
        g_b[0] = [0.25 * (np.random.rand() * 2 - 1),
                  0.25 * (np.random.rand() * 2 - 1)]

        for t in range(n_obs - 1):
            g_b[t + 1] = (g_b[t] + g_b_lambda * g_b_mu[t + 1]) / (1 +
                                                                  g_b_lambda)
            var = sigma ** 2 / (1 + g_b_lambda)
            g_b[t + 1] += (np.random.randn(1, 2) * np.sqrt(var)).reshape(2,)

            error_b = g_b[t + 1] - p_b[t]
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

        p.append(np.hstack((p_g, p_b)))
        g.append(g_b)

    return p, g


def batch_generator_pad(arrays, batch_size):
    """Minibatch generator over one dataset of shape
    (nobservations, ndimensions)
    """
    size = len(arrays)
    n_batch = math.ceil(size / batch_size)
    np.random.shuffle(arrays)
    start = 0
    while True:
        batches = []
        for _ in range(n_batch):
            stop = start + batch_size
            diff = stop - size
            if diff <= 0:
                batch = arrays[start:stop]
                start = stop
            else:
                batch = arrays[start:]
            batch = data_pad(batch)
            batches.append(batch)
        yield batches

def data_pad(array):
    max_len = np.max([len(a) for a in array])
    return np.array([np.pad(a, ((0, max_len-len(a)), (0, 0)),'edge') for a in array])


def batch_generator_pad(arrays, batch_size):
    """Minibatch generator over one dataset of shape
    (nobservations, ndimensions)
    """
    size = len(arrays)
    n_batch = math.ceil(size / batch_size)
    np.random.shuffle(arrays)
    start = 0
    while True:
        batches = []
        for _ in range(n_batch):
            stop = start + batch_size
            diff = stop - size
            if diff <= 0:
                batch = arrays[start:stop]
                start = stop
            else:
                batch = arrays[start:]
            batch = data_pad(batch)
            batches.append(batch)
        yield batches


def data_pad(array):
    max_len = np.max([len(a) for a in array])
    return np.array([np.pad(a, ((0, max_len-len(a)), (0, 0)),
                            'edge') for a in array])
    # return np.asarray([np.pad(a, ((0, max_len-len(a)), (0, 0)), 'constant',
    #                           constant_values=0) for a in batch])


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

import argparse
from tf_gbds.utils import *
# from tf_gbds.prep_GAN import prep_GAN_data, prep_injection_conditions
import sys
import os
from os.path import expanduser
import tensorflow as tf
# from tensorflow.python.client import timeline
import numpy as np
import time
import tf_gbds.GenerativeModel_GMM_Edward as G
import tf_gbds.RecognitionModel_Edward as R
import edward as ed
from edward import KLqp

# import pickle as pkl
# export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

MODEL_DIR = 'model_gmm'
# MAX_SESSIONS = 10
# SESSION_TYPE = 'recording'
# SESSION_INDEX_DIR = ''
DATA_DIR = ''

REC_LAG = 10
REC_NLAYERS = 3
REC_HIDDEN_DIM = 25

GEN_NLAYERS = 3
GEN_HIDDEN_DIM = 64
K = 20
C = 8

ADD_ACCEL = False
EPS_INIT = 1e-5
EPS_PENALTY = None
SIGMA_INIT = 1e-5
SIGMA_PENALTY = None
# model class allows for having 2 boundaries with different penalties,
# but we later found that unnecessary, so the CLI only takes on penalty.
# We left the possibility for 2 penalties in the model class just in case
# it may be useful on a different dataset/task
GOAL_BOUNDARY = 1.0
GOAL_BOUND_PENALTY = None

SEED = 1234
TRAIN_OPTIMIZER = 'Adam'
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2000
BATCH_SIZE = 128


flags = tf.app.flags()

flags.Define_string('model_type', 'KLqp',
                    'Type of model to build {KLqp, HMM}')
flags.Define_string('model_dir', MODEL_DIR,
                    'Directory where the model is saved')
flags.Define_integer('max_sessions', MAX_SESSIONS,
                     'Maximum number of sessions to load')
flags.Define_string('session_type', SESSION_TYPE,
                    'Type of data session')
flags.Define_string('session_index_dir', SESSION_INDEX_DIR,
                    'Directory of session index file')
flags.Define_string('data_dir', DATA_DIR, 'Directory of data file')

flags.Define_integer('rec_lag', REC_LAG, 'Number of previous timepoints \
                     to include as input to recognition model')
flags.Define_integer('rec_nlayers', REC_NLAYERS, 'Number of layers in \
                     recognition model neural networks')
flags.Define_integer('rec_hidden_dim', REC_HIDDEN_DIM,
                     'Number of hidden units in each (dense) layer of \
                     recognition model neural networks')

flags.Define_integer('gen_nlayers', GEN_NLAYERS, 'Number of layers in \
                     generative model neural networks')
flags.Define_integer('gen_hidden_dim', GEN_HIDDEN_DIM,
                     'number of hidden units in each (dense) layer of \
                     generative model neural networks')
flags.Define_integer('K', K, 'Number of sub-strategies (components of GMM)')
flags.Define_integer('C', C, 'Number of highest-level strategies')

flags.Define_boolean('add_accel', ADD_ACCEL,
                     'Should acceleration be added to states?')
flags.Define_float('eps_init', EPS_INIT,
                   'Initial value of control signal noise')
flags.Define_float('eps_penalty', EPS_PENALTY,
                   'Penalty on control signal noise')
flags.Define_float('sigma_init', SIGMA_INIT,
                   'Initial value of goal state noise')
flags.Define_float('sigma_penalty', SIGMA_PENALTY,
                   'Penalty on goal state noise')
flags.Define_float('goal_bound', GOAL_BOUNDARY, 'Goal state boundaries')
flags.Define_float('goal_bound_penalty', GOAL_BOUND_PENALTY,
                   'Penalty for goal states escaping boundaries')

flags.Define_integer('seed', SEED, 'Random seed')
flags.Define_string('optimizer', TRAIN_OPTIMIZER, 'Training optimizer')
flags.Define_float('learning_rate', LEARNING_RATE, 'Initial learning rate')
flags.Define_integer('num_epochs', NUM_EPOCHS,
                     'Number of iterations through the full training set')
flags.Define_integer('batch_size', BATCH_SIZE, 'Size of mini-batch')

FLAGS = flags.FLAGS

def build_hyperparameter_dict(flags):
    d = {}

    d['model_dir'] = flags.model_dir
    d['max_sessions'] = flags.max_sessions
    d['session_type'] = flags.session_type
    d['session_index_dir'] = flags.session_index_dir
    d['data_dir'] = flags.data_dir

    d['rec_lag'] = flags.rec_lag
    d['rec_nlayers'] = flags.rec_nlayers
    d['rec_hidden_dim'] = flags.rec_hidden_dim

    d['gen_nlayers'] = flags.gen_nlayers
    d['gen_hidden_dim'] = flags.gen_hidden_dim
    d['K'] = flags.K
    d['C'] = flags.C

    d['add_accel'] = flags.add_accel
    d['eps_init'] = flags.eps_init
    d['eps_penalty'] = flags.eps_penalty
    d['sigma_init'] = flags.sigma_init
    d['sigma_penalty'] = flags.sigma_penalty
    d['goal_bound'] = flags.goal_bound
    d['goal_bound_penalty'] = flags.goal_bound_penalty

    d['seed'] = flags.seed
    d['optimizer'] = flags.optimizer
    d['learning_rate'] = flags.learning_rate
    d['num_epochs'] = flags.num_epochs
    d['batch_size'] = flags.batch_size

    return d

class hps_dict_to_obj(dict):
  """Helper class allowing us to access hps dictionary more easily."""

  def __getattr__(self, key):
    if key in self:
      return self[key]
    else:
      assert False, ("%s does not exist." % key)
  def __setattr__(self, key, value):
    self[key] = value

def load_data:

def model_setup:

def train:

def write_model_samples:

def write_model_parameters:

def main(_):
    d = build_hyperparameter_dict(FLAGS)
    hps = hps_dict_to_obj(d)    # hyper-parameters
    model_type = FLAGS.model_type

# if __name__ == "__main__":
#     tf.app.run()
    
def initialize_PID_params(type, yDim):
    PID_params = dict()
    PID_params['unc_Kp'] = tf.Variable(initial_value=np.zeros((yDim, 1)),
                                       name='unc_Kp_' + type,
                                       dtype=tf.float32)
    PID_params['unc_Ki'] = tf.Variable(initial_value=np.zeros((yDim, 1)),
                                       name='unc_Ki_' + type,
                                       dtype=tf.float32)
    PID_params['unc_Kd'] = tf.Variable(initial_value=np.zeros((yDim, 1)),
                                       name='unc_Kd_' + type,
                                       dtype=tf.float32)
    PID_params['unc_eps'] = tf.Variable(
        initial_value=-11.513 * np.ones((1, yDim)), name='unc_eps_' + type,
        dtype=tf.float32)
    return PID_params

def initialize_Dyn_params(type, RecognitionParams):
    Dyn_params = dict()
    Dyn_params['A'] = tf.Variable(RecognitionParams['A'], name='A_' + type,
                                  dtype=tf.float32)
    Dyn_params['QinvChol'] = tf.Variable(RecognitionParams['QinvChol'],
                                         name='QinvChol_' + type,
                                         dtype=tf.float32)
    Dyn_params['Q0invChol'] = tf.Variable(RecognitionParams['Q0invChol'],
                                          name='Q0invChol' + type,
                                          dtype=tf.float32)
    return Dyn_params

def run_model(**kwargs):
    if not os.path.exists(outname):
        os.makedirs(outname)

    data, goals = gen_data(n_trial=2000, n_obs=100, Kp=0.8, Ki=0.4, Kd=0.2)

    np.random.seed(seed)  # set seed for consistent train/val split
    train_data = []
    val_data = []
    val_goals = []
    for (trial_data, trial_goals) in zip(data, goals):
        if np.random.rand() <= 0.85:
            train_data.append(trial_data)
        else:
            val_data.append(trial_data)
            val_goals.append(trial_goals)
    np.save(outname + '/train_data', train_data)
    np.save(outname + '/val_data', val_data)
    np.save(outname + '/val_goals', val_goals)

    yCols_goalie = [0]
    yCols_ball = [1, 2]

    obs_dim_g = len(yCols_goalie)
    obs_dim_b = len(yCols_ball)
    obs_dim = obs_dim_g + obs_dim_b

    # No CLI arguments for these bc no longer being used, but left just in case
    penalty_Q = None
    PKLparams = None
    row_sparse = False
    add_pklayers = False

    vel = get_max_velocities(train_data, val_data)
    ntrials = len(train_data)
    val_ntrials = len(val_data)

    with tf.name_scope('params'):
        with tf.name_scope('rec_control_params'):
            rec_params_u = get_rec_params_GBDS(
                obs_dim, rec_lag, nlayers_rec, hidden_dim_rec, penalty_Q,
                PKLparams, 'u')
        with tf.name_scope('rec_goal_params'):
            rec_params_g = get_rec_params_GBDS(
                obs_dim, rec_lag, nlayers_rec, hidden_dim_rec, penalty_Q,
                PKLparams, 'g')

        with tf.name_scope('gen_goalie_params'):
            gen_params_goalie = get_gen_params_GBDS_GMM(
                obs_dim_g, obs_dim, add_accel, yCols_goalie, nlayers_gen,
                hidden_dim_gen, K, C, B, PKLparams, vel, penalty_eps,
                penalty_sigma, boundaries_g, penalty_g, 'goalie')
        with tf.name_scope('gen_ball_params'):
            gen_params_ball = get_gen_params_GBDS_GMM(
                obs_dim_b, obs_dim, add_accel, yCols_ball, nlayers_gen,
                hidden_dim_gen, K, C, B, PKLparams, vel, penalty_eps,
                penalty_sigma, boundaries_g, penalty_g, 'ball')

        Y = tf.placeholder(tf.float32, shape=(None, None), name='data')
        yDim_goalie = len(yCols_goalie)
        yDim_ball = len(yCols_ball)
        yDim = tf.shape(Y_ph)[-1]
        xDim = yDim

        PID_params_goalie = initialize_PID_params('goalie', yDim_goalie)
        PID_params_ball = initialize_PID_params('ball', yDim_ball)
        Dyn_params_u = initialize_Dyn_params('u', rec_params_u)
        Dyn_params_g = initialize_Dyn_params('g', rec_params_g)

    with tf.name_scope('model_setup'):
        with tf.name_scope('gen_g'):
            g = G.GBDS_g_all(gen_params_goalie, gen_params_ball, yDim, Y_ph,
                             value=tf.ones([tf.shape(Y)[0],yDim]))
        with tf.name_scope('gen_u'):
            u = G.GBDS_u_all(gen_params_goalie, gen_params_ball, g, Y_ph, yDim,
                             PID_params_goalie, PID_params_ball,
                             value=tf.ones([tf.shape(Y)[0],yDim]))

        with tf.name_scope('rec_g'):
            # qg = R.SmoothingLDSTimeSeries(rec_params_g, Y, xDim, yDim, Dyn_params_g)
            qg = R.SmoothingPastLDSTimeSeries(rec_params_g, Y_ph, xDim, yDim,
                                              Dyn_params_g, ntrials)
        with tf.name_scope('rec_u'):
            # qu = R.SmoothingLDSTimeSeries(rec_params_u, Y, xDim, yDim, Dyn_params_g)
            qu = R.SmoothingPastLDSTimeSeries(rec_params_u, Y_ph, xDim, yDim,
                                              Dyn_params_u, ntrials)

        # Y_pred_t = Y_(t-1)+max_vel*tanh(u_t) ,where Y_pred_0 = Y_0
        Y_pred = tf.concat([tf.expand_dims(Y[0],0),
                            (Y[:-1] + tf.reshape(vel, [1, yDim]) *
                                tf.clip_by_value(u[1:], -1, 1))], 0)

    print('--------------Generative Params----------------')
    # if penalty_eps is not None:
    #     print('Penalty on control signal noise, epsilon (Generative): %i' % penalty_eps)
    # if penalty_sigma is not None:
    #     print('Penalty on goal state noise, sigma (Generative): %i' % penalty_sigma)
    # if penalty_g[0] is not None:
    #     print('Penalty on goal state leaving boundary 1 (Generative): %i' % penalty_g[0])
    # if penalty_g[1] is not None:
    #     print('Penalty on goal state leaving boundary 2 (Generative): %i' % penalty_g[1])
    print('Num Substrategies: %i' % K)
    print('Num Highest-level Strategies: %i' % C)
    print('--------------Recognition Params---------------')
    print('Num Layers (VILDS recognition): %i' % nlayers_rec)
    print('Hidden Dims (VILDS recognition): %i' % hidden_dim_rec)
    print('Input lag (VILDS recognition): %i' % rec_lag)
    # sys.stdout.flush()

    # set up an iterator over our training data
    batch_size = 1
    data_iter_vb = DatasetTrialIndexIterator(train_data, randomize=True)
    # data_iter_vd = DatasetTrialIndexIterator(val_data, randomize=True)

    n_iter_per_epoch = ntrials // batch_size

    val_costs = []
    ctrl_cost = []

    print('Setting up VB model...')
    var_list = g.getParams() + u.getParams() + qg.getParams() + qu.getParams()

    with tf.name_scope('goal_samples'):
        q_g_mean = tf.squeeze(qg.postX, 2,
                              name='q_g_mean')
        q_g = tf.squeeze(qg.sample(30), name='q_g')
    with tf.name_scope('control_signal_samples'):
        q_U_mean = tf.squeeze(qu.postX, 2,
                              name='q_U_mean')
        q_U = tf.squeeze(qu.sample(30), name='q_U')

    with tf.name_scope('train_KLqp'):   
        inference = KLqp({g:qg, u:qu}, data={Y_pred: Y})
        # inference.initialize(var_list=var_list,
        #                      optimizer=tf.train.AdamOptimizer(learning_rate),
        #                      logdir = "/home/qiankuang/Documents/projects/model_saved")
        inference.initialize(var_list=var_list,
                             optimizer=tf.train.AdamOptimizer(learning_rate))

    tf.summary.scalar('Kp_x_ball', u.u_ball.Kp[0, 0])
    tf.summary.scalar('Kp_y_ball', u.u_ball.Kp[1, 0])
    tf.summary.scalar('Ki_x_ball', u.u_ball.Ki[0, 0])
    tf.summary.scalar('Ki_y_ball', u.u_ball.Ki[1, 0])
    tf.summary.scalar('Kd_x_ball', u.u_ball.Kd[0, 0])
    tf.summary.scalar('Kd_y_ball', u.u_ball.Kd[1, 0])

    summary_op = tf.summary.merge_all()

    sess = ed.get_session()
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter(outname, sess.graph)

    print('---> Training model')
    # sys.stdout.flush()
    for ie in range(n_epochs):
        # Iterate over the training data for the specified number of epochs
        print('--> entering epoch %i' % (ie + 1))
        start_train = time.time()
        avg_loss = 0.0
        val_loss = 0.0

        for data in data_iter_vb:
            info_dict = inference.update(feed_dict = {Y: data})
            avg_loss += info_dict['loss']

        avg_loss = avg_loss / batch_size / n_iter_per_epoch
        ctrl_cost.append(avg_loss)

        for i in range(len(val_data)):
            val_loss += sess.run(inference.loss, feed_dict = {Y: val_data[i]})
        val_loss = val_loss / val_ntrials
        val_costs.append(val_loss)

        train_summary = sess.run(summary_op)
        # inference.print_progress(info_dict)
        train_writer.add_summary(train_summary)

        np.save(outname + '/train_costs', ctrl_cost)
        np.save(outname + '/val_costs', val_costs)

        print("training loss: {:0.3f}".format(avg_loss))
        print("validation set loss: {:0.3f}".format(val_loss))
        print('Epoch %i takes %0.3f s' % ((ie + 1), (time.time() - start_train)))

        if (ie + 1) % 20 == 0:
            print('----> Predicting control signals and goals')
            ctrl_post_mean = []
            ctrl_post_samp = []
            goal_post_mean = []

            for i in range(len(val_data)):
                ctrl_post_mean.append(
                    q_U_mean.eval(feed_dict={Y: val_data[i]}))
                ctrl_post_samp.append(
                    q_U.eval(feed_dict={Y: val_data[i]}))
                goal_post_mean.append(
                    q_g_mean.eval(feed_dict={Y: val_data[i]}))

            np.save(outname + '/ctrl_post_mean_step_%s' % (ie + 1),
                    ctrl_post_mean)
            np.save(outname + '/ctrl_post_samp_step_%s' % (ie + 1),
                    ctrl_post_samp)
            np.save(outname + '/goal_post_mean_step_%s' % (ie + 1),
                    goal_post_mean)

    train_writer.close()
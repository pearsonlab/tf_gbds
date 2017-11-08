from tf_gbds.utils import *
# from tf_gbds.prep_GAN import prep_GAN_data, prep_injection_conditions
import os
from os.path import expanduser
import tensorflow as tf
# from tensorflow.python.client import timeline
import numpy as np
import time
import tf_gbds.GenerativeModel_GMM_Edward as G
import tf_gbds.RecognitionModel_Edward as R
import edward as ed

# import pickle as pkl
# export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

MODEL_DIR = 'model_gmm'
# MAX_SESSIONS = 10
# SESSION_TYPE = 'recording'
# SESSION_INDEX_DIR = ''
DATA_DIR = ''
SYNTHETIC_DATA = True
SAVE_POSTERIOR = True

P1_DIM = 1
P2_DIM = 2

REC_LAG = 10
REC_NLAYERS = 3
REC_HIDDEN_DIM = 25

GEN_NLAYERS = 3
GEN_HIDDEN_DIM = 64
K = 8

ADD_ACCEL = False
CLIP = True
CLIP_RANGE = 1.
CLIP_TOL = 1e-5
ETA = 1e-6
EPS_INIT = 1e-5
EPS_TRAINABLE = False
EPS_PENALTY = None
SIGMA_INIT = 1e-5
SIGMA_TRAINABLE = False
SIGMA_PENALTY = None
# model class allows for having 2 boundaries with different penalties,
# but we later found that unnecessary, so the CLI only takes on penalty.
# We left the possibility for 2 penalties in the model class just in case
# it may be useful on a different dataset/task
GOAL_BOUNDARY = 1.0
GOAL_BOUND_PENALTY = None

SEED = 1234
TRAIN_RATIO = 0.85
TRAIN_OPTIMIZER = 'Adam'
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2000
BATCH_SIZE = 128
NUM_SAMPLES = 30


flags = tf.app.flags

flags.DEFINE_string('model_type', 'VB_KLqp',
                    'Type of model to build {VB_KLqp, HMM')
flags.DEFINE_string('model_dir', MODEL_DIR,
                    'Directory where the model is saved')
# flags.DEFINE_integer('max_sessions', MAX_SESSIONS,
#                      'Maximum number of sessions to load')
# flags.DEFINE_string('session_type', SESSION_TYPE,
#                     'Type of data session')
# flags.DEFINE_string('session_index_dir', SESSION_INDEX_DIR,
#                     'Directory of session index file')
flags.DEFINE_string('data_dir', DATA_DIR, 'Directory of data file')
flags.DEFINE_boolean('synthetic_data', SYNTHETIC_DATA,
                     'Is the model trained on synthetic data?')
flags.DEFINE_boolean('save_posterior', SAVE_POSTERIOR, 'Will posterior \
                     distributions be retrieved after training?')

flags.DEFINE_integer('p1_dim', P1_DIM,
                     'Number of data dimensions corresponding to player 1')
flags.DEFINE_integer('p2_dim', P2_DIM,
                     'Number of data dimensions corresponding to player 2')

flags.DEFINE_integer('rec_lag', REC_LAG, 'Number of previous timepoints \
                     to include as input to recognition model')
flags.DEFINE_integer('rec_nlayers', REC_NLAYERS, 'Number of layers in \
                     recognition model neural networks')
flags.DEFINE_integer('rec_hidden_dim', REC_HIDDEN_DIM,
                     'Number of hidden units in each (dense) layer of \
                     recognition model neural networks')

flags.DEFINE_integer('gen_nlayers', GEN_NLAYERS, 'Number of layers in \
                     generative model neural networks')
flags.DEFINE_integer('gen_hidden_dim', GEN_HIDDEN_DIM,
                     'Number of hidden units in each (dense) layer of \
                     generative model neural networks')
flags.DEFINE_integer('K', K, 'Number of sub-strategies (components of GMM)')

flags.DEFINE_boolean('add_accel', ADD_ACCEL,
                     'Should acceleration be added to states?')
flags.DEFINE_boolean('clip', CLIP, 'Is the control signal censored?')
flags.DEFINE_float('clip_range', CLIP_RANGE,
                   'The range beyond which control signals are censored')
flags.DEFINE_float('clip_tol', CLIP_TOL,
                   'The tolerance of signal censoring')
flags.DEFINE_float('eta', ETA, 'The scale of control signal noise')
flags.DEFINE_float('eps_init', EPS_INIT,
                   'Initial value of control signal variance')
flags.DEFINE_boolean('eps_trainable', EPS_TRAINABLE,
                     'Is epsilon trainable?')
flags.DEFINE_float('eps_penalty', EPS_PENALTY,
                   'Penalty on control signal variance')
flags.DEFINE_float('sigma_init', SIGMA_INIT,
                   'Initial value of goal state variance')
flags.DEFINE_boolean('sigma_trainable', SIGMA_TRAINABLE,
                     'Is sigma trainable?')
flags.DEFINE_float('sigma_penalty', SIGMA_PENALTY,
                   'Penalty on goal state variance')
flags.DEFINE_float('goal_bound', GOAL_BOUNDARY, 'Goal state boundaries')
flags.DEFINE_float('goal_bound_penalty', GOAL_BOUND_PENALTY,
                   'Penalty for goal states escaping boundaries')

flags.DEFINE_integer('seed', SEED, 'Random seed')
flags.DEFINE_float('train_ratio', TRAIN_RATIO,
                   'The proportion of data used for training')
flags.DEFINE_string('optimizer', TRAIN_OPTIMIZER, 'Training optimizer')
flags.DEFINE_float('learning_rate', LEARNING_RATE, 'Initial learning rate')
flags.DEFINE_integer('num_epochs', NUM_EPOCHS,
                     'Number of iterations through the full training set')
flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of mini-batch')
flags.DEFINE_integer('num_samples', NUM_SAMPLES,
                     'Number of samples drawn from posterior distributions')

FLAGS = flags.FLAGS

def build_hyperparameter_dict(flags):
    d = {}

    d['model_type'] = flags.model_type
    d['model_dir'] = flags.model_dir
    # d['max_sessions'] = flags.max_sessions
    # d['session_type'] = flags.session_type
    # d['session_index_dir'] = flags.session_index_dir
    d['data_dir'] = flags.data_dir
    d['synthetic_data'] = flags.synthetic_data
    d['save_posterior'] = flags.save_posterior

    d['p1_dim'] = flags.p1_dim
    d['p2_dim'] = flags.p2_dim

    d['rec_lag'] = flags.rec_lag
    d['rec_nlayers'] = flags.rec_nlayers
    d['rec_hidden_dim'] = flags.rec_hidden_dim

    d['gen_nlayers'] = flags.gen_nlayers
    d['gen_hidden_dim'] = flags.gen_hidden_dim
    d['K'] = flags.K

    d['add_accel'] = flags.add_accel
    d['clip'] = flags.clip
    d['clip_range'] = flags.clip_range
    d['clip_tol'] = flags.clip_tol
    d['eta'] = flags.eta
    d['eps_init'] = flags.eps_init
    d['eps_trainable'] = flags.eps_trainable
    d['eps_penalty'] = flags.eps_penalty
    d['sigma_init'] = flags.sigma_init
    d['sigma_trainable'] = flags.sigma_trainable
    d['sigma_penalty'] = flags.sigma_penalty
    d['goal_bound'] = flags.goal_bound
    d['goal_bound_penalty'] = flags.goal_bound_penalty

    d['seed'] = flags.seed
    d['train_ratio'] = flags.train_ratio
    d['opt'] = flags.optimizer
    d['learning_rate'] = flags.learning_rate
    d['n_epochs'] = flags.num_epochs
    d['B'] = flags.batch_size
    d['n_samples'] = flags.num_samples

    return d

class hps_dict_to_obj(dict):
    '''Helper class allowing us to access hps dictionary more easily.'''
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            assert False, ('%s does not exist.' % key)
    def __setattr__(self, key, value):
        self[key] = value

def load_data(hps):
    if hps.synthetic_data:
        data, goals = gen_data(
            n_trial=2000, n_obs=100, Kp=0.8, Ki=0.4, Kd=0.2)
        np.random.seed(hps.seed)  # set seed for consistent train/val split
        train_data = []
        val_data = []
        val_goals = []
        for (trial_data, trial_goals) in zip(data, goals):
            if np.random.rand() <= hps.train_ratio:
                train_data.append(trial_data)
            else:
                val_data.append(trial_data)
                val_goals.append(trial_goals)
        np.save(hps.model_dir + '/train_data', train_data)
        np.save(hps.model_dir + '/val_data', val_data)
        np.save(hps.model_dir + '/val_goals', val_goals)
    elif hps.data_dir is not None:
        goals = None
        train_data, val_data = load_pk_data(hps)  # to be edited
    else:
        raise Exception('Data must be provided (either real or synthetic)')

    return train_data, val_data

def run_model(model_type, hps):
    if not os.path.exists(hps.model_dir):
        os.makedirs(hps.model_dir)

    train_data, val_data = load_data(hps)
    val_ntrials = len(val_data)
    val_data = np.array(val_data)
    print('Data loaded.')

    with tf.name_scope('params'):
        with tf.name_scope('columns'):
            p1_cols = np.arange(hps.p1_dim)
            p2_cols = hps.p1_dim + np.arange(hps.p2_dim)
            total_dim = hps.p1_dim + hps.p2_dim

        # No CLI arguments for these bc no longer being used,
        # but left just in case
        penalty_Q = None
        PKLparams = None
        row_sparse = False
        add_pklayers = False

        vel = get_max_velocities(train_data, val_data)
        train_ntrials = len(train_data)
        val_ntrials = len(val_data)

        with tf.name_scope('rec_control_params'):
            rec_params_u = get_rec_params_GBDS(
                total_dim, hps.rec_lag, hps.rec_nlayers, hps.rec_hidden_dim,
                penalty_Q, PKLparams, name='U')
            Dyn_params_u = init_Dyn_params('U', rec_params_u)
        with tf.name_scope('rec_goal_params'):
            rec_params_g = get_rec_params_GBDS(
                total_dim, hps.rec_lag, hps.rec_nlayers, hps.rec_hidden_dim,
                penalty_Q, PKLparams, name='G')
            Dyn_params_g = init_Dyn_params('G', rec_params_g)

        with tf.name_scope('gen_goalie_params'):
            gen_params_goalie = get_gen_params_GBDS_GMM(
                hps.p1_dim, total_dim, hps.add_accel, p1_cols, hps.gen_nlayers,
                hps.gen_hidden_dim, hps.K, PKLparams, vel,
                hps.eps_penalty, hps.sigma_penalty,
                hps.goal_bound, hps.goal_bound_penalty,
                hps.clip, hps.clip_range, hps.clip_tol, hps.eta, name='Goalie')
            PID_params_goalie = init_PID_params('Goalie', hps.p1_dim)
        with tf.name_scope('gen_ball_params'):
            gen_params_ball = get_gen_params_GBDS_GMM(
                hps.p2_dim, total_dim, hps.add_accel, p2_cols, hps.gen_nlayers,
                hps.gen_hidden_dim, hps.K, PKLparams, vel,
                hps.eps_penalty, hps.sigma_penalty,
                hps.goal_bound, hps.goal_bound_penalty,
                hps.clip, hps.clip_range, hps.clip_tol, hps.eta, name='Ball')
            PID_params_ball = init_PID_params('Ball', hps.p2_dim)

    with tf.name_scope('model_setup'):
        Y_ph = tf.placeholder(tf.float32, shape=(None, None, total_dim),
                              name='data')

        with tf.name_scope('gen_g'):
            p_G = G.GBDS_g_all(gen_params_goalie, gen_params_ball, total_dim,
                               Y_ph, value=tf.zeros_like(Y_ph))
        with tf.name_scope('gen_u'):
            p_U = G.GBDS_u_all(gen_params_goalie, gen_params_ball, p_G, Y_ph,
                               total_dim, PID_params_goalie, PID_params_ball,
                               value=tf.zeros_like(Y_ph))

        with tf.name_scope('rec_g'):
            q_G = R.SmoothingPastLDSTimeSeries(rec_params_g, Y_ph, total_dim,
                                               total_dim, Dyn_params_g,
                                               train_ntrials)
        with tf.name_scope('rec_u'):
            q_U = R.SmoothingPastLDSTimeSeries(rec_params_u, Y_ph, total_dim,
                                               total_dim, Dyn_params_u,
                                               train_ntrials)

        with tf.name_scope('obs'):
            # Y_pred_t = Y_(t-1)+max_vel*tanh(u_t) ,where Y_pred_0 = Y_0
            if hps.clip:
                Y = tf.concat([tf.expand_dims(Y_ph[:, 0], 1), (Y_ph[:, :-1] +
                               (tf.reshape(vel, [1, total_dim]) *
                                tf.clip_by_value(p_U[:, 1:], -hps.clip_range,
                                                 hps.clip_range)))], 1,
                              name='Y')
            else:
                Y = tf.concat([tf.expand_dims(Y_ph[:, 0], 1), (Y_ph[:, :-1] +
                               (tf.reshape(vel, [1, total_dim]) *
                                p_U[:, 1:]))], 1,
                              name='Y')

    print('--------------Generative Params----------------')
    if hps.eps_penalty is not None:
        print('Penalty on control signal variance, epsilon (Generative): %i' % hps.eps_penalty)
    if hps.sigma_penalty is not None:
        print('Penalty on goal state variance, sigma (Generative): %i' % hps.sigma_penalty)
    if hps.goal_bound_penalty is not None:
        print('Penalty on goal state leaving boundary (Generative): %i' % hps.goal_bound_penalty)
    print('Number of GMM components: %i' % hps.K)
    print('--------------Recognition Params---------------')
    print('Num Layers (VILDS recognition): %i' % hps.rec_nlayers)
    print('Hidden Dims (VILDS recognition): %i' % hps.rec_hidden_dim)
    print('Input lag (VILDS recognition): %i' % hps.rec_lag)

    if hps.save_posterior:
        with tf.name_scope('posterior'):
            with tf.name_scope('goal'):
                q_G_mean = tf.squeeze(q_G.postX, -1, name='mean')
                q_G_samp = tf.identity(q_G.sample(hps.n_samples),
                                       name='samples')
            with tf.name_scope('control_signal'):
                q_U_mean = tf.squeeze(q_U.postX, -1, name='mean')
                q_U_samp = tf.identity(q_U.sample(hps.n_samples),
                                       name='samples')
            with tf.name_scope('GMM_goalie'):
                GMM_mu, GMM_lambda, GMM_w, _ = p_G.goalie.get_preds(
                    Y_ph, training=True, post_g=q_G.sample(1))
            with tf.name_scope('GMM_ball'):
                GMM_mu, GMM_lambda, GMM_w, _ = p_G.ball.get_preds(
                    Y_ph, training=True, post_g=q_G.sample(1))

    if model_type == 'VB_KLqp':
        batches = next(batch_generator(train_data, hps.B))
        n_batches = math.ceil(len(train_data) / hps.B)
        var_list = (p_G.getParams() + p_U.getParams() +
                    q_G.getParams() + q_U.getParams())
        if hps.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(hps.learning_rate)

        inference = ed.KLqp({p_G:q_G, p_U:q_U}, data={Y: Y_ph})
        inference.initialize(n_iter=hps.n_epochs * n_batches,
                             var_list=var_list,
                             optimizer=optimizer,
                             logdir=hps.model_dir + '/log')
        sess = ed.get_session()
        tf.global_variables_initializer().run()

        for i in range(hps.n_epochs):
            for batch in batches:
                info_dict = inference.update({Y_ph: batch})
                inference.print_progress(info_dict)

            val_loss = sess.run(inference.loss,
                                feed_dict={Y_ph: val_data})
            print('\n', 'Validation set loss after epoch %i: %.3f' %
                  (i + 1, val_loss / val_ntrials))

        saver = tf.train.Saver()
        saver.save(sess, hps.model_dir + '/saved_model')

# def write_model_samples:

# def write_model_parameters:

def main(_):
    d = build_hyperparameter_dict(FLAGS)
    hps = hps_dict_to_obj(d)  # hyper-parameters
    model_type = FLAGS.model_type
    run_model(model_type, hps)

if __name__ == "__main__":
    tf.app.run()

# def run_model(**kwargs):

#     # set up an iterator over our training data
#     batch_size = 1
#     data_iter_vb = DatasetTrialIndexIterator(train_data, randomize=True)
#     # data_iter_vd = DatasetTrialIndexIterator(val_data, randomize=True)

#     n_iter_per_epoch = ntrials // batch_size

#     val_costs = []
#     ctrl_cost = []

#     print('Setting up VB model...')
#     var_list = g.getParams() + u.getParams() + qg.getParams() + qu.getParams()

#     with tf.name_scope('goal_samples'):
#         q_g_mean = tf.squeeze(qg.postX, 2,
#                               name='q_g_mean')
#         q_g = tf.squeeze(qg.sample(30), name='q_g')
#     with tf.name_scope('control_signal_samples'):
#         q_U_mean = tf.squeeze(qu.postX, 2,
#                               name='q_U_mean')
#         q_U = tf.squeeze(qu.sample(30), name='q_U')

#     with tf.name_scope('train_KLqp'):   
#         inference = KLqp({g:qg, u:qu}, data={Y_pred: Y})
#         # inference.initialize(var_list=var_list,
#         #                      optimizer=tf.train.AdamOptimizer(learning_rate),
#         #                      logdir = "/home/qiankuang/Documents/projects/model_saved")
#         inference.initialize(var_list=var_list,
#                              optimizer=tf.train.AdamOptimizer(learning_rate))

#     tf.summary.scalar('Kp_x_ball', u.u_ball.Kp[0, 0])
#     tf.summary.scalar('Kp_y_ball', u.u_ball.Kp[1, 0])
#     tf.summary.scalar('Ki_x_ball', u.u_ball.Ki[0, 0])
#     tf.summary.scalar('Ki_y_ball', u.u_ball.Ki[1, 0])
#     tf.summary.scalar('Kd_x_ball', u.u_ball.Kd[0, 0])
#     tf.summary.scalar('Kd_y_ball', u.u_ball.Kd[1, 0])

#     summary_op = tf.summary.merge_all()

#     sess = ed.get_session()
#     tf.global_variables_initializer().run()
#     train_writer = tf.summary.FileWriter(outname, sess.graph)

#     print('---> Training model')
#     # sys.stdout.flush()
#     for ie in range(n_epochs):
#         # Iterate over the training data for the specified number of epochs
#         print('--> entering epoch %i' % (ie + 1))
#         start_train = time.time()
#         avg_loss = 0.0
#         val_loss = 0.0

#         for data in data_iter_vb:
#             info_dict = inference.update(feed_dict = {Y: data})
#             avg_loss += info_dict['loss']

#         avg_loss = avg_loss / batch_size / n_iter_per_epoch
#         ctrl_cost.append(avg_loss)

#         for i in range(len(val_data)):
#             val_loss += sess.run(inference.loss, feed_dict = {Y: val_data[i]})
#         val_loss = val_loss / val_ntrials
#         val_costs.append(val_loss)

#         train_summary = sess.run(summary_op)
#         # inference.print_progress(info_dict)
#         train_writer.add_summary(train_summary)

#         np.save(outname + '/train_costs', ctrl_cost)
#         np.save(outname + '/val_costs', val_costs)

#         print("training loss: {:0.3f}".format(avg_loss))
#         print("validation set loss: {:0.3f}".format(val_loss))
#         print('Epoch %i takes %0.3f s' % ((ie + 1), (time.time() - start_train)))

#         if (ie + 1) % 20 == 0:
#             print('----> Predicting control signals and goals')
#             ctrl_post_mean = []
#             ctrl_post_samp = []
#             goal_post_mean = []

#             for i in range(len(val_data)):
#                 ctrl_post_mean.append(
#                     q_U_mean.eval(feed_dict={Y: val_data[i]}))
#                 ctrl_post_samp.append(
#                     q_U.eval(feed_dict={Y: val_data[i]}))
#                 goal_post_mean.append(
#                     q_g_mean.eval(feed_dict={Y: val_data[i]}))

#             np.save(outname + '/ctrl_post_mean_step_%s' % (ie + 1),
#                     ctrl_post_mean)
#             np.save(outname + '/ctrl_post_samp_step_%s' % (ie + 1),
#                     ctrl_post_samp)
#             np.save(outname + '/goal_post_mean_step_%s' % (ie + 1),
#                     goal_post_mean)

#     train_writer.close()
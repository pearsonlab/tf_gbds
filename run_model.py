import os
import tensorflow as tf
import numpy as np
import math
import edward as ed
import tf_gbds.GenerativeModel as G
import tf_gbds.RecognitionModel as R
from tf_gbds.utils import (load_data, get_max_velocities,
                           get_rec_params_GBDS, get_gen_params_GBDS_GMM,
                           batch_generator, batch_generator_pad, pad_batch,
                           KLqp_profile, add_summary)
import time
from tensorflow.python.client import timeline

# export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH


MODEL_DIR = 'model_gmm'
DATA_DIR = ''
SYNTHETIC_DATA = False
SAVE_POSTERIOR = True
LOAD_SAVED_MODEL = False
SAVED_MODEL_DIR = ''
PROFILE = False

P1_DIM = 1
P2_DIM = 2

REC_LAG = 10
REC_NLAYERS = 3
REC_HIDDEN_DIM = 25

GEN_NLAYERS = 3
GEN_HIDDEN_DIM = 64
K = 8

ADD_ACCEL = False
LATENT_CONTROL = True
CLIP = True
CLIP_RANGE = 1.
CLIP_TOL = 1e-5
ETA = 1e-6
CONTROL_ERROR_PENALTY = None
EPS_INIT = 1e-10
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
NUM_EPOCHS = 500
BATCH_SIZE = 1
NUM_SAMPLES = 1
NUM_POSTERIOR_SAMPLES = 30
MAX_CKPT_TO_KEEP = 5
FREQUENCY_SAVING_CKPT = 5
FREQUENCY_VAL_LOSS = 5

flags = tf.app.flags

flags.DEFINE_string('model_type', 'VI_KLqp',
                    'Type of model to build {VI_KLqp, HMM}')
flags.DEFINE_string('model_dir', MODEL_DIR,
                    'Directory where the model is saved')
flags.DEFINE_string('data_dir', DATA_DIR, 'Directory of data file')
flags.DEFINE_boolean('synthetic_data', SYNTHETIC_DATA,
                     'Is the model trained on synthetic data?')
flags.DEFINE_boolean('save_posterior', SAVE_POSTERIOR, 'Will posterior \
                     samples be retrieved after training?')
flags.DEFINE_boolean('load_saved_model', LOAD_SAVED_MODEL, 'Is the model \
                     restored from a checkpoint?')
flags.DEFINE_string('saved_model_dir', SAVED_MODEL_DIR,
                    'Directory where the model to be restored is saved')
flags.DEFINE_boolean('profile', PROFILE, 'Is the model being profiled? \
                     Use absolute path for hps.model_dir if profiling')
flags.DEFINE_string('device_type', 'CPU',
                    'The device where the model is trained {CPU, GPU}')

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
flags.DEFINE_boolean('latent_u', LATENT_CONTROL, 'Is observed control signal \
                     modeled as a noisy version of latent control?')
flags.DEFINE_boolean('clip', CLIP, 'Is the control signal censored?')
flags.DEFINE_float('clip_range', CLIP_RANGE,
                   'The range beyond which control signals are censored')
flags.DEFINE_float('clip_tol', CLIP_TOL,
                   'The tolerance of signal censoring')
flags.DEFINE_float('eta', ETA, 'The scale of control signal noise')
flags.DEFINE_float('ctrl_error_penalty', CONTROL_ERROR_PENALTY,
                   'Penalty on control error (input to PID controll model)')
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
flags.DEFINE_integer('num_samples', NUM_SAMPLES, 'Number of posterior \
                     samples for calculating stochastic gradients')
flags.DEFINE_integer('num_posterior_samples', NUM_POSTERIOR_SAMPLES,
                     'Number of samples drawn from posterior distributions')
flags.DEFINE_integer('max_ckpt_to_keep', MAX_CKPT_TO_KEEP,
                     'maximum number of checkpoint to keep in the directory')
flags.DEFINE_integer('frequency_saving_ckpt', FREQUENCY_SAVING_CKPT,
                     'frequency of saving checkpoint')
flags.DEFINE_integer('frequency_val_loss', FREQUENCY_VAL_LOSS, 'frequency of \
                     computing validation loss')

FLAGS = flags.FLAGS


def build_hyperparameter_dict(flags):
    d = {}

    d['model_type'] = flags.model_type
    d['model_dir'] = flags.model_dir
    d['data_dir'] = flags.data_dir
    d['synthetic_data'] = flags.synthetic_data
    d['save_posterior'] = flags.save_posterior
    d['load_saved_model'] = flags.load_saved_model
    d['saved_model_dir'] = flags.saved_model_dir
    d['profile'] = flags.profile
    d['device_type'] = flags.device_type

    d['p1_dim'] = flags.p1_dim
    d['p2_dim'] = flags.p2_dim

    d['rec_lag'] = flags.rec_lag
    d['rec_nlayers'] = flags.rec_nlayers
    d['rec_hidden_dim'] = flags.rec_hidden_dim

    d['gen_nlayers'] = flags.gen_nlayers
    d['gen_hidden_dim'] = flags.gen_hidden_dim
    d['K'] = flags.K

    d['add_accel'] = flags.add_accel
    d['latent_u'] = flags.latent_u
    d['clip'] = flags.clip
    d['clip_range'] = flags.clip_range
    d['clip_tol'] = flags.clip_tol
    d['eta'] = flags.eta
    d['ctrl_error_penalty'] = flags.ctrl_error_penalty
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
    d['n_post_samples'] = flags.num_posterior_samples
    d['max_ckpt_to_keep'] = flags.max_ckpt_to_keep
    d['frequency_saving_ckpt'] = flags.frequency_saving_ckpt
    d['frequency_val_loss'] = flags.frequency_val_loss

    return d


class hps_dict_to_obj(dict):
    '''Helper class allowing us to access hps dictionary more easily.
    '''
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            assert False, ('%s does not exist.' % key)

    def __setattr__(self, key, value):
        self[key] = value


def run_model(hps):
    '''Train GBDS model on real data
    '''
    if not os.path.exists(hps.model_dir):
        os.makedirs(hps.model_dir)

    (train_data, train_conds, train_ctrls, val_data,
        val_conds, val_ctrls) = load_data(hps)
    extra_conds_present = train_conds is not None
    ctrl_obs_present = train_ctrls is not None

    val_data = pad_batch(val_data, mode='edge')
    if ctrl_obs_present:
        val_ctrls = pad_batch(val_ctrls, mode='zero')
    print('Data loaded.')

    # time1 = time.time()
    with tf.name_scope('params'):
        with tf.name_scope('columns'):
            p1_cols = np.arange(hps.p1_dim)
            p2_cols = hps.p1_dim + np.arange(hps.p2_dim)
            obs_dim = hps.p1_dim + hps.p2_dim

        # No CLI arguments for these bc no longer being used,
        # but left just in case
        penalty_Q = None
        PKLparams = None

        vel = get_max_velocities(train_data, val_data)
        train_ntrials = len(train_data)
        val_ntrials = len(val_data)

        if extra_conds_present:
            extra_dim = train_conds.shape[-1]
        else:
            extra_dim = 0

        # Initialize all of the parameters in the model
        with tf.name_scope('rec_control_params'):
            rec_params_u = get_rec_params_GBDS(
                obs_dim, extra_dim, hps.rec_lag, hps.rec_nlayers,
                hps.rec_hidden_dim, penalty_Q, PKLparams, name='U')
        with tf.name_scope('rec_goal_params'):
            rec_params_g = get_rec_params_GBDS(
                obs_dim, extra_dim, hps.rec_lag, hps.rec_nlayers,
                hps.rec_hidden_dim, penalty_Q, PKLparams, name='G')

        with tf.name_scope('gen_goalie_params'):
            gen_params_goalie = get_gen_params_GBDS_GMM(
                hps.p1_dim, obs_dim, extra_dim, hps.add_accel, p1_cols,
                hps.gen_nlayers, hps.gen_hidden_dim, hps.K, PKLparams, vel,
                hps.ctrl_error_penalty, hps.eps_penalty, hps.sigma_penalty,
                hps.goal_bound, hps.goal_bound_penalty, hps.latent_u,
                hps.clip, hps.clip_range, hps.clip_tol, hps.eta, name='Goalie')
        with tf.name_scope('gen_ball_params'):
            gen_params_ball = get_gen_params_GBDS_GMM(
                hps.p2_dim, obs_dim, extra_dim, hps.add_accel, p2_cols,
                hps.gen_nlayers, hps.gen_hidden_dim, hps.K, PKLparams, vel,
                hps.ctrl_error_penalty, hps.eps_penalty, hps.sigma_penalty,
                hps.goal_bound, hps.goal_bound_penalty, hps.latent_u,
                hps.clip, hps.clip_range, hps.clip_tol, hps.eta, name='Ball')

    with tf.name_scope('model_setup'):
        # Creat the placeholder to input data
        Y_ph = tf.placeholder(tf.float32, shape=(None, None, obs_dim),
                              name='data')
        extra_conds_ph = tf.placeholder(tf.float32, shape=(None, extra_dim),
                                        name='extra_conds')
        ctrl_obs_ph = tf.placeholder(tf.float32, shape=(None, None, obs_dim),
                                     name='ctrl_obs')
        # Generate real goal and control signal
        with tf.name_scope('gen_G'):
            if extra_conds_present:
                if ctrl_obs_present:
                    p_G = G.GBDS_g_all(
                        gen_params_goalie, gen_params_ball, obs_dim, Y_ph,
                        extra_conds_ph, ctrl_obs_ph, name='p_G',
                        value=tf.zeros_like(Y_ph))
                else:
                    p_G = G.GBDS_g_all(
                        gen_params_goalie, gen_params_ball, obs_dim, Y_ph,
                        extra_conds_ph, None, name='p_G',
                        value=tf.zeros_like(Y_ph))
            elif ctrl_obs_present:
                p_G = G.GBDS_g_all(
                    gen_params_goalie, gen_params_ball, obs_dim, Y_ph, None,
                    ctrl_obs_ph, name='p_G',
                    value=tf.zeros_like(Y_ph))
            else:
                p_G = G.GBDS_g_all(
                    gen_params_goalie, gen_params_ball, obs_dim, Y_ph, None,
                    None, name='p_G', value=tf.zeros_like(Y_ph))

        with tf.name_scope('gen_U'):
            if hps.latent_u:
                if ctrl_obs_present:
                    p_U = G.GBDS_u_all(gen_params_goalie, gen_params_ball,
                                       p_G, Y_ph, ctrl_obs_ph, obs_dim,
                                       name='p_U', value=tf.zeros_like(Y_ph))
                else:
                    p_U = G.GBDS_u_all(gen_params_goalie, gen_params_ball,
                                       p_G, Y_ph, None, obs_dim,
                                       name='p_U', value=tf.zeros_like(Y_ph))

        # Generate posterior goal and control signal
        with tf.name_scope('rec_G'):
            if extra_conds_present:
                q_G = R.SmoothingPastLDSTimeSeries(
                    rec_params_g, Y_ph, extra_conds_ph, obs_dim, obs_dim,
                    name='q_G')
            else:
                q_G = R.SmoothingPastLDSTimeSeries(
                    rec_params_g, Y_ph, None, obs_dim, obs_dim, name='q_G')

        with tf.name_scope('rec_U'):
            if hps.latent_u:
                if extra_conds_present:
                    q_U = R.SmoothingPastLDSTimeSeries(
                        rec_params_u, Y_ph, extra_conds_ph, obs_dim, obs_dim,
                        name='q_U')
                else:
                    q_U = R.SmoothingPastLDSTimeSeries(
                        rec_params_u, Y_ph, None, obs_dim, obs_dim, name='q_U')

        # Generate real state based on control signal and velocility
        with tf.name_scope('observations'):
            with tf.name_scope('control_signals'):
                if hps.latent_u:
                    U = p_U
                else:
                    U = tf.concat([p_G.goalie.u, p_G.ball.u], -1)
            with tf.name_scope('trajectories'):
                if hps.clip:
                    Y = tf.concat([tf.expand_dims(Y_ph[:, 0], 1),
                                   (Y_ph[:, :-1] + (tf.reshape(
                                        vel, [1, obs_dim]) *
                                    tf.clip_by_value(
                                        U[:, :-1], -hps.clip_range,
                                        hps.clip_range)))], 1, name='Y')
                else:
                    Y = tf.concat([tf.expand_dims(Y_ph[:, 0], 1),
                                   (Y_ph[:, :-1] + (tf.reshape(
                                        vel, [1, obs_dim]) *
                                    U[:, :-1]))], 1, name='Y')

    print('--------------Generative Params----------------')
    if hps.eps_penalty is not None:
        print('Penalty on control signal variance, epsilon (Generative): %i'
              % hps.eps_penalty)
    if hps.sigma_penalty is not None:
        print('Penalty on goal state variance, sigma (Generative): %i'
              % hps.sigma_penalty)
    if hps.goal_bound_penalty is not None:
        print('Penalty on goal state leaving boundary (Generative): %i'
              % hps.goal_bound_penalty)
    if hps.ctrl_error_penalty is not None:
        print('Penalty on large control error (Generative): %i'
              % hps.ctrl_error_penalty)
    print('Num GMM components: %i' % hps.K)

    print('--------------Recognition Params---------------')
    print('Num Layers (VILDS recognition): %i' % hps.rec_nlayers)
    print('Hidden Dims (VILDS recognition): %i' % hps.rec_hidden_dim)
    print('Input Lag (VILDS recognition): %i' % hps.rec_lag)

    with tf.name_scope('PID_params_summary'):
        PID_summary_key = tf.get_default_graph().unique_name(
            'PID_params_summary')

        if hps.latent_u:
            U_goalie = p_U.goalie
            U_ball = p_U.ball
        else:
            U_goalie = p_G.goalie.u
            U_ball = p_G.ball.u

        Kp_goalie = tf.summary.scalar('PID_params/goalie/Kp',
                                      U_goalie.Kp[0, 0],
                                      collections=PID_summary_key)
        # Ki_goalie = tf.summary.scalar('PID_params/goalie/Ki',
        #                               U_goalie.Ki[0, 0],
        #                               collections=PID_summary_key)
        # Kd_goalie = tf.summary.scalar('PID_params/goalie/Kd',
        #                               U_goalie.Kd[0, 0],
        #                               collections=PID_summary_key)
        Kp_ball_x = tf.summary.scalar('PID_params/ball_x/Kp',
                                      U_ball.Kp[0, 0],
                                      collections=PID_summary_key)
        # Ki_ball_x = tf.summary.scalar('PID_params/ball_x/Ki',
        #                               U_ball.Ki[0, 0],
        #                               collections=PID_summary_key)
        # Kd_ball_x = tf.summary.scalar('PID_params/ball_x/Kd',
        #                               U_ball.Kd[0, 0],
        #                               collections=PID_summary_key)
        Kp_ball_y = tf.summary.scalar('PID_params/ball_y/Kp',
                                      U_ball.Kp[1, 0],
                                      collections=PID_summary_key)
        # Ki_ball_y = tf.summary.scalar('PID_params/ball_y/Ki',
        #                               U_ball.Ki[1, 0],
        #                               collections=PID_summary_key)
        # Kd_ball_y = tf.summary.scalar('PID_params/ball_y/Kd',
        #                               U_ball.Kd[1, 0],
        #                               collections=PID_summary_key)

        # PID_summary = tf.summary.merge([Kp_goalie, Ki_goalie, Kd_goalie,
        #                                 Kp_ball_x, Ki_ball_x, Kd_ball_x,
        #                                 Kp_ball_y, Ki_ball_y, Kd_ball_y],
        #                                collections=PID_summary_key,
        #                                name='PID_params_summary')
        PID_summary = tf.summary.merge([Kp_goalie, Kp_ball_x, Kp_ball_y],
                                       collections=PID_summary_key,
                                       name='PID_params_summary')

    if hps.save_posterior:
        with tf.name_scope('posterior'):
            with tf.name_scope('goal'):
                q_G_mean = tf.squeeze(q_G.postX, -1, name='mean')
                q_G_samp = tf.identity(q_G.sample(hps.n_post_samples),
                                       name='samples')
            with tf.name_scope('control_signal'):
                if hps.latent_u:
                    q_U_mean = tf.squeeze(q_U.postX, -1, name='mean')
                    q_U_samp = tf.identity(q_U.sample(hps.n_post_samples),
                                           name='samples')
            with tf.name_scope('GMM_goalie'):
                GMM_mu, GMM_lambda, GMM_w, _ = p_G.goalie.get_preds(
                    Y_ph, training=True,
                    post_g=tf.gather(q_G.sample(), p1_cols, axis=-1))
                g0_mu = tf.identity(p_G.goalie.g0_mu, name='g0_mu')
                g0_lambda = tf.identity(p_G.goalie.g0_lambda,
                                        name='g0_lambda')
                g0_w = tf.identity(p_G.goalie.g0_w, name='g0_w')
            with tf.name_scope('GMM_ball'):
                GMM_mu, GMM_lambda, GMM_w, _ = p_G.ball.get_preds(
                    Y_ph, training=True,
                    post_g=tf.gather(q_G.sample(), p2_cols, axis=-1))
                g0_mu = tf.identity(p_G.ball.g0_mu, name='g0_mu')
                g0_lambda = tf.identity(p_G.ball.g0_lambda, name='g0_lambda')
                g0_w = tf.identity(p_G.ball.g0_w, name='g0_w')

    # Calculate variational inference using Edward KLqp function
    if hps.model_type == 'VI_KLqp':
        n_batches = math.ceil(train_ntrials / hps.B)
        if hps.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(hps.learning_rate)

        if hps.latent_u:
            var_list = (p_G.getParams() + p_U.getParams() +
                        q_G.getParams() + q_U.getParams())
            latent_vars = {p_G: q_G, p_U: q_U}
        else:
            var_list = (p_G.getParams() + q_G.getParams())
            latent_vars = {p_G: q_G}

        if hps.profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            inference = KLqp_profile(options, run_metadata,
                                     latent_vars=latent_vars, data={Y: Y_ph})
        else:
            inference = ed.KLqp(latent_vars=latent_vars, data={Y: Y_ph})

        inference.initialize(n_iter=hps.n_epochs * n_batches,
                             n_samples=hps.n_samples,
                             scale={Y: 1 / hps.B},
                             var_list=var_list,
                             optimizer=optimizer,
                             logdir=hps.model_dir + '/log')

        sess = ed.get_session()
        tf.global_variables_initializer().run()

        lowest_ev_cost = np.Inf
        seso_saver = tf.train.Saver(tf.global_variables(),
                                    max_to_keep=hps.max_ckpt_to_keep)
        lve_saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=2)

        if hps.load_saved_model:
            seso_saver.restore(sess, hps.saved_model_dir)
            print('Model restored from ' + hps.saved_model_dir)

        # time2 = time.time()
        # print('Model setup took %.3f s.' % (time2 - time1))

        for i in range(hps.n_epochs):
            if hps.synthetic_data:
                batches = next(batch_generator(train_data, hps.B))
            else:
                batches, conds, ctrls = next(batch_generator_pad(
                    train_data, hps.B, train_conds, train_ctrls))

            if extra_conds_present and ctrl_obs_present:
                for batch, cond, ctrl in zip(batches, conds, ctrls):
                    feed_dict = {Y_ph: batch, extra_conds_ph: cond,
                                 ctrl_obs_ph: ctrl}
                    info_dict = inference.update(feed_dict=feed_dict)
                    add_summary(PID_summary, inference, sess, feed_dict,
                                info_dict['t'])
                    inference.print_progress(info_dict)
            elif extra_conds_present:
                for batch, cond in zip(batches, conds):
                    feed_dict = {Y_ph: batch, extra_conds_ph: cond}
                    info_dict = inference.update(feed_dict=feed_dict)
                    add_summary(PID_summary, inference, sess, feed_dict,
                                info_dict['t'])
                    inference.print_progress(info_dict)
            elif ctrl_obs_present:
                for batch, ctrl in zip(batches, ctrls):
                    feed_dict = {Y_ph: batch, ctrl_obs_ph: ctrl}
                    info_dict = inference.update(feed_dict=feed_dict)
                    add_summary(PID_summary, inference, sess, feed_dict,
                                info_dict['t'])
                    inference.print_progress(info_dict)
            else:
                for batch in batches:
                    feed_dict = {Y_ph: batch}
                    info_dict = inference.update(feed_dict=feed_dict)
                    add_summary(PID_summary, inference, sess, feed_dict,
                                info_dict['t'])
                    inference.print_progress(info_dict)

            if (i + 1) % hps.frequency_saving_ckpt == 0:
                seso_saver.save(sess, hps.model_dir + '/saved_model',
                                global_step=(i + 1),
                                latest_filename='checkpoint')

            if (i + 1) % hps.frequency_val_loss == 0:
                if extra_conds_present and ctrl_obs_present:
                    val_loss = sess.run(
                        inference.loss,
                        feed_dict={Y_ph: val_data, extra_conds_ph: val_conds,
                                   ctrl_obs_ph: val_ctrls})
                elif extra_conds_present:
                    val_loss = sess.run(
                        inference.loss,
                        feed_dict={Y_ph: val_data, extra_conds_ph: val_conds})
                elif ctrl_obs_present:
                    val_loss = sess.run(
                        inference.loss,
                        feed_dict={Y_ph: val_data, ctrl_obs_ph: val_ctrls})
                else:
                    val_loss = sess.run(
                        inference.loss, feed_dict={Y_ph: val_data})

                print('\n', 'Validation loss after epoch %i: %.3f' %
                      (i + 1, val_loss * hps.B / val_ntrials))

                if val_loss * hps.B / val_ntrials < lowest_ev_cost:
                    print('Saving model with lowest validation loss...')
                    lowest_ev_cost = val_loss * hps.B / val_ntrials
                    lve_saver.save(sess, hps.model_dir + '/saved_model_lve',
                                   global_step=(i + 1),
                                   latest_filename='checkpoint_lve',
                                   write_meta_graph=False)

        if hps.profile:
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # use absolute path for hps.model_dir
            with open(hps.model_dir + '/timeline_01_step_%d.json' %
                      (i + 1), 'w') as f:
                f.write(chrome_trace)
                f.close()

        seso_saver.save(sess, hps.model_dir + '/final_model')

        print('Model has been saved. Training ends.')

        # time3 = time.time()
        # print('Model training took %.3f s.' % (time3 - time2))


def main(_):
    d = build_hyperparameter_dict(FLAGS)
    hps = hps_dict_to_obj(d)  # hyper-parameters

    if hps.device_type == 'CPU':
        with tf.device('cpu:0'):
            run_model(hps)
    elif hps.device_type == 'GPU':
        run_model(hps)
    else:
        raise Exception('Device type not recognized.')


if __name__ == '__main__':
    tf.app.run()

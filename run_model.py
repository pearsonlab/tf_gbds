import os
import tensorflow as tf
import numpy as np
import math
import edward as ed
import tf_gbds.GenerativeModel as G
import tf_gbds.RecognitionModel as R
from tf_gbds.utils import (load_data, get_max_velocities,
                           get_vel, get_accel, get_agent_params,
                           batch_generator, batch_generator_pad, pad_batch,
                           KLqp_profile, add_summary)
import time
from tensorflow.python.client import timeline


# default flag values

MODEL_DIR = 'new_model'
DATA_DIR = None
SYNTHETIC_DATA = False
SAVE_POSTERIOR = True
LOAD_SAVED_MODEL = False
SAVED_MODEL_DIR = None
PROFILE = False

N_AGENTS = 2
AGENT_NAME = 'goalie,kicker'
AGENT_COLUMN = '0;1,2'
OBSERVE_DIM = 3
ADD_ACCEL = False

GMM_K = 8
GEN_N_LAYERS = 3
GEN_HIDDEN_DIM = 64
REC_LAG = 10
REC_N_LAYERS = 3
REC_HIDDEN_DIM = 25

SIGMA = 1e-3
SIGMA_TRAINABLE = False
GOAL_BOUNDARY_L = -1.
GOAL_BOUNDARY_U = 1.
GOAL_BOUNDARY_PENALTY = None

CONTROL_RESIDUAL = 1e-5
CONTROL_RESIDUAL_PENALTY = 1e8
CONTROL_ERROR_PENALTY = None
LATENT_CONTROL = False
CLIP = False
CLIP_RANGE_L = -1.
CLIP_RANGE_U = 1.
CLIP_TOL = 1e-5
CLIP_PENALTY = 1e8

SEED = 1234
VAL_SET = True
TRAIN_RATIO = 0.85
OPTIMIZER = 'Adam'
LEARNING_RATE = 1e-3
N_EPOCHS = 500
BATCH_SIZE = 1
N_VI_SAMPLES = 1
N_POSTERIOR_SAMPLES = 30
MAX_CKPT = 5
FREQ_CKPT = 5
FREQ_VAL_LOSS = 5


flags = tf.app.flags

flags.DEFINE_string('model_dir', MODEL_DIR,
                    'Directory where the model is saved')
flags.DEFINE_string('data_dir', DATA_DIR, 'Directory of input data file')
flags.DEFINE_boolean('syn_data', SYNTHETIC_DATA,
                     'Is the model trained on synthetic dataset')
flags.DEFINE_boolean('save_posterior', SAVE_POSTERIOR, 'Will posterior \
                     samples be saved after training?')
flags.DEFINE_boolean('load_saved_model', LOAD_SAVED_MODEL, 'Is the model \
                     restored from an existing checkpoint')
flags.DEFINE_string('saved_model_dir', SAVED_MODEL_DIR,
                    'Directory where the model to be restored is saved')
flags.DEFINE_boolean('profile', PROFILE, 'Is the model being profiled \
                     (use absolute path for FLAGS.model_dir if profiling)')
flags.DEFINE_string('device', 'CPU',
                    'The device where the model is trained (CPU or GPU)')

flags.DEFINE_integer('n_agents', N_AGENTS, 'Number of agents in the model')
flags.DEFINE_string('agent_name', AGENT_NAME, 'Name of each agent \
                    (separated by ',')')
flags.DEFINE_string('agent_col', AGENT_COLUMN, 'Columns of data \
                    corresponding to each agent (separated by ';' and ',')')
flags.DEFINE_integer('obs_dim', OBSERVE_DIM, 'Number of observed dimensions')
flags.DEFINE_boolean('add_accel', ADD_ACCEL,
                     'Is acceleration included in state')

flags.DEFINE_integer('GMM_K', GMM_K, 'Number of components in GMM')
flags.DEFINE_integer('gen_n_layers', GEN_N_LAYERS, 'Number of layers in \
                     neural networks (generative model)')
flags.DEFINE_integer('gen_hidden_dim', GEN_HIDDEN_DIM,
                     'Number of hidden units in each dense layer of \
                     neural networks (generative model)')
flags.DEFINE_integer('rec_lag', REC_LAG, 'Number of previous timepoints \
                     included as input to recognition model')
flags.DEFINE_integer('rec_nlayers', REC_N_LAYERS, 'Number of layers in \
                     neural networks (recognition model)')
flags.DEFINE_integer('rec_hidden_dim', REC_HIDDEN_DIM,
                     'Number of hidden units in each dense layer of \
                     neural networks (recognition model)')

flags.DEFINE_float('sigma', SIGMA, 'Initial value of goal state variance')
flags.DEFINE_boolean('sigma_trainable', SIGMA_TRAINABLE, 'Is sigma trainable')
flags.DEFINE_float('g_lb', GOAL_BOUNDARY_L, 'Goal state lower boundary')
flags.DEFINE_float('g_ub', GOAL_BOUNDARY_U, 'Goal state upper boundary')
flags.DEFINE_float('g_bounds_pen', GOAL_BOUNDARY_PENALTY,
                   'Penalty on goal states escaping boundaries')

flags.DEFINE_float('u_res', CONTROL_RESIDUAL,
                   'Tolerance of control signal residual')
flags.DEFINE_float('u_res_pen', CONTROL_RESIDUAL_PENALTY,
                   'Penalty on control signal residual')
flags.DEFINE_float('u_error_pen', CONTROL_ERROR_PENALTY,
                   'Penalty on large control errors (input to PID control)')
flags.DEFINE_boolean('latent_u', LATENT_CONTROL, 'Is the true control signal \
                     modeled as latent variable')
flags.DEFINE_boolean('clip', CLIP, 'Is the observed control signal censored')
flags.DEFINE_float('clip_lb', CLIP_RANGE_L,
                   'Control signal censoring lower bound')
flags.DEFINE_float('clip_ub', CLIP_RANGE_U,
                   'Control signal censoring upper bound')
flags.DEFINE_float('clip_tol', CLIP_TOL,
                   'Tolerance of control signal censoring')
flags.DEFINE_float('clip_pen', CLIP_PENALTY,
                   'Penalty on control signal censoring')

flags.DEFINE_integer('seed', SEED, 'Random seed for numpy functions')
flags.DEFINE_boolean('val', VAL_SET, 'Is dataset split into training and \
                     validation sets')
flags.DEFINE_float('train_ratio', TRAIN_RATIO,
                   'Proportion of data used for training')
flags.DEFINE_string('opt', OPTIMIZER, 'Gradient descent optimizer')
flags.DEFINE_float('lr', LEARNING_RATE, 'Initial learning rate')
flags.DEFINE_integer('n_epochs', N_EPOCHS, 'Number of iterations algorithm \
                    runs through the training set')
flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of mini-batches')
flags.DEFINE_integer('n_samp', N_VI_SAMPLES, 'Number of samples drawn \
                     for gradient estimation')
flags.DEFINE_integer('n_post_samp', N_POSTERIOR_SAMPLES, 'Number of samples \
                     from posterior distributions to draw and save')
flags.DEFINE_integer('max_ckpt', MAX_CKPT,
                     'Maximum number of checkpoints to keep in the directory')
flags.DEFINE_integer('freq_ckpt', FREQ_CKPT, 'Frequency of saving \
                     checkpoints to the directory')
flags.DEFINE_integer('freq_val_loss', FREQ_VAL_LOSS, 'Frequency of \
                     evaluating loss of validation set')

FLAGS = flags.FLAGS


def run_model(FLAGS):
    '''Train GBDS model on real data
    '''
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    (train_data, train_conds, train_ctrls,
        val_data, val_conds, val_ctrls) = load_data(FLAGS)
    extra_conds_present = train_conds is not None
    ctrl_obs_present = train_ctrls is not None

    if val_data is not None:
        val_data = pad_batch(val_data, mode='edge')
        if ctrl_obs_present:
            val_ctrls = pad_batch(val_ctrls, mode='zero')

    print('Data loaded.')

    agent_name = FLAGS.agent_name.split(',')
    assert len(agent_name) == FLAGS.n_agents, 'length of name list %i do not \
        match number of agents %i' % (len(agent_name), FLAGS.n_agents)

    agent_col = [[int(c) for c in a.split(',')]
                 for a in FLAGS.agent_col.split(';')]
    assert len(agent_col) == FLAGS.n_agents, 'length of column list %i \
        do not match number of agents %i' % (len(agent_col), FLAGS.n_agents)

    agent_dim = [len(a) for a in agent_col]
    assert sum(agent_dim) <= FLAGS.obs_dim, 'modeling dimensions more than \
        observed: %i > %i' % (sum(agent_dim), FLAGS.obs_dim)

    train_ntrials = len(train_data)
    if val_data is not None:
        val_ntrials = len(val_data)

    vel = get_max_velocities([train_data, val_data], FLAGS.obs_dim)

    if FLAGS.add_accel:
        state_dim = obs_dim * 3
        get_state = get_accel
    else:
        state_dim = obs_dim * 2
        get_state = get_vel

    if extra_conds_present:
        extra_dim = train_conds.shape[-1]
    else:
        extra_dim = 0

    # No CLI arguments for these bc no longer being used,
    # but left just in case
    penalty_Q = None
    PKLparams = None

    if FLAGS.clip:
        clip_range = [FLAGS.clip_lb, FLAGS.clip_ub]
    else:
        clip_range = None

    agent_params = []
    for agent in agents:
        params = get_agent_params(
            name, agent_dim, agent_cols, obs_dim, state_dim, extra_dim,
            FLAGS.gen_nlayers, FLAGS.gen_hidden_dim, FLAGS.K, FLAGS.sigma_init,
            FLAGS.goal_bound, FLAGS.goal_bound_penalty, PKLparams, vel,
            FLAGS.latent_u, FLAGS.rec_lag, FLAGS.rec_nlayers, FLAGS.rec_hidden_dim,
            # penalty_Q, FLAGS.penalty_ctrl, FLAGS.ctrl_residual_tolerance,
            FLAGS.clip, clip_range, FLAGS.clip_tol, FLAGS.eta,
            FLAGS.ctrl_error_penalty)
        agent_params.append(params)

    with tf.name_scope('inputs'):
        Y = tf.placeholder(tf.float32, shape=(None, None, obs_dim),
                           name='trajectories')
        inputs = {'trajectories': Y, 'states': get_state(Y)}

        if extra_conds_present:
            extra_conds = tf.placeholder(tf.float32, shape=(None, extra_dim),
                                         name='extra_conditions')
            inputs.update({'extra_conds': extra_conds})
        if ctrl_obs_present:
            ctrl_obs = tf.placeholder(tf.float32, shape=(None, None, obs_dim),
                                      name='observed_control')
            inputs.update({'ctrl_obs': ctrl_obs})

    model = game_model(agent_params, inputs, name='penaltykick')

    #     # Initialize all of the parameters in the model
    #     with tf.name_scope('rec_control_params'):
    #         rec_params_u = get_rec_params_GBDS(
    #             obs_dim, extra_dim, FLAGS.rec_lag, FLAGS.rec_nlayers,
    #             FLAGS.rec_hidden_dim, penalty_Q, PKLparams, name='U')
    #     with tf.name_scope('rec_goal_params'):
    #         rec_params_g = get_rec_params_GBDS(
    #             obs_dim, extra_dim, FLAGS.rec_lag, FLAGS.rec_nlayers,
    #             FLAGS.rec_hidden_dim, penalty_Q, PKLparams, name='G')

    #     with tf.name_scope('gen_goalie_params'):
    #         gen_params_goalie = get_gen_params_GBDS_GMM(
    #             FLAGS.p1_dim, obs_dim, extra_dim, FLAGS.add_accel, p1_cols,
    #             FLAGS.gen_nlayers, FLAGS.gen_hidden_dim, FLAGS.K, PKLparams, vel,
    #             FLAGS.ctrl_error_penalty, FLAGS.eps_penalty, FLAGS.sigma_penalty,
    #             FLAGS.goal_bound, FLAGS.goal_bound_penalty, FLAGS.latent_u,
    #             FLAGS.clip, FLAGS.clip_range, FLAGS.clip_tol, FLAGS.eta, name='Goalie')
    #     with tf.name_scope('gen_ball_params'):
    #         gen_params_ball = get_gen_params_GBDS_GMM(
    #             FLAGS.p2_dim, obs_dim, extra_dim, FLAGS.add_accel, p2_cols,
    #             FLAGS.gen_nlayers, FLAGS.gen_hidden_dim, FLAGS.K, PKLparams, vel,
    #             FLAGS.ctrl_error_penalty, FLAGS.eps_penalty, FLAGS.sigma_penalty,
    #             FLAGS.goal_bound, FLAGS.goal_bound_penalty, FLAGS.latent_u,
    #             FLAGS.clip, FLAGS.clip_range, FLAGS.clip_tol, FLAGS.eta, name='Ball')

    # with tf.name_scope('model_setup'):
    #     # Creat the placeholder to input data
    #     Y_ph = tf.placeholder(tf.float32, shape=(None, None, obs_dim),
    #                           name='data')
    #     extra_conds_ph = tf.placeholder(tf.float32, shape=(None, extra_dim),
    #                                     name='extra_conds')
    #     ctrl_obs_ph = tf.placeholder(tf.float32, shape=(None, None, obs_dim),
    #                                  name='ctrl_obs')
    #     # Generate real goal and control signal
    #     with tf.name_scope('gen_G'):
    #         if extra_conds_present:
    #             if ctrl_obs_present:
    #                 p_G = G.GBDS_g_all(
    #                     gen_params_goalie, gen_params_ball, obs_dim, Y_ph,
    #                     extra_conds_ph, ctrl_obs_ph, name='p_G',
    #                     value=tf.zeros_like(Y_ph))
    #             else:
    #                 p_G = G.GBDS_g_all(
    #                     gen_params_goalie, gen_params_ball, obs_dim, Y_ph,
    #                     extra_conds_ph, None, name='p_G',
    #                     value=tf.zeros_like(Y_ph))
    #         elif ctrl_obs_present:
    #             p_G = G.GBDS_g_all(
    #                 gen_params_goalie, gen_params_ball, obs_dim, Y_ph, None,
    #                 ctrl_obs_ph, name='p_G', value=tf.zeros_like(Y_ph))
    #         else:
    #             p_G = G.GBDS_g_all(
    #                 gen_params_goalie, gen_params_ball, obs_dim, Y_ph, None,
    #                 None, name='p_G', value=tf.zeros_like(Y_ph))

    #     with tf.name_scope('gen_U'):
    #         if FLAGS.latent_u:
    #             if ctrl_obs_present:
    #                 p_U = G.GBDS_u_all(gen_params_goalie, gen_params_ball,
    #                                    p_G, Y_ph, ctrl_obs_ph, obs_dim,
    #                                    name='p_U', value=tf.zeros_like(Y_ph))
    #             else:
    #                 p_U = G.GBDS_u_all(gen_params_goalie, gen_params_ball,
    #                                    p_G, Y_ph, None, obs_dim,
    #                                    name='p_U', value=tf.zeros_like(Y_ph))

    #     # Generate posterior goal and control signal

    #     with tf.name_scope('rec_G'):
    #         if extra_conds_present:
    #             q_G = R.SmoothingPastLDSTimeSeries(
    #                 rec_params_g, Y_ph, extra_conds_ph, obs_dim, obs_dim,
    #                 name='q_G')
    #         else:
    #             q_G = R.SmoothingPastLDSTimeSeries(
    #                 rec_params_g, Y_ph, None, obs_dim, obs_dim, name='q_G')

    #     with tf.name_scope('rec_U'):
    #         if FLAGS.latent_u:
    #             if extra_conds_present:
    #                 q_U = R.SmoothingPastLDSTimeSeries(
    #                     rec_params_u, Y_ph, extra_conds_ph, obs_dim, obs_dim,
    #                     name='q_U')
    #             else:
    #                 q_U = R.SmoothingPastLDSTimeSeries(
    #                     rec_params_u, Y_ph, None, obs_dim, obs_dim,
    #                     name='q_U')

    #     # Generate real state based on control signal and velocility
    #     with tf.name_scope('observations'):
    #         with tf.name_scope('control_signals'):
    #             if FLAGS.latent_u:
    #                 U = p_U
    #             else:
    #                 U = tf.concat([p_G.goalie.u, p_G.ball.u], -1)
    #         with tf.name_scope('trajectories'):
    #             if FLAGS.clip:
    #                 Y = tf.concat([tf.expand_dims(Y_ph[:, 0], 1),
    #                                (Y_ph[:, :-1] + (tf.reshape(
    #                                     vel, [1, obs_dim]) *
    #                                 tf.clip_by_value(
    #                                     U[:, :-1], -FLAGS.clip_range,
    #                                     FLAGS.clip_range)))], 1, name='Y')
    #             else:
    #                 Y = tf.concat([tf.expand_dims(Y_ph[:, 0], 1),
    #                                (Y_ph[:, :-1] + (tf.reshape(
    #                                     vel, [1, obs_dim]) *
    #                                 U[:, :-1]))], 1, name='Y')

    print('--------------Generative Parameters--------------')
    print('Number of GMM components: %i' % FLAGS.K)
    print('Number of layers in neural networks: %i' % FLAGS.gen_nlayers)
    print('Dimensions of hidden layers: %i' % FLAGS.gen_hidden_dim)
    if FLAGS.goal_bound_penalty is not None:
        print('Penalty on goal states leaving boundary (Generative): %i'
              % FLAGS.goal_bound_penalty)
    if FLAGS.ctrl_error_penalty is not None:
        print('Penalty on large control errors (Generative): %i'
              % FLAGS.ctrl_error_penalty)

    print('--------------Recognition Parameters--------------')
    print('Number of layers in neural networks: %i' % FLAGS.rec_nlayers)
    print('Dimensions of hidden layers: %i' % FLAGS.rec_hidden_dim)
    print('Lag of input: %i' % FLAGS.rec_lag)

    with tf.name_scope('PID_params_summary'):
        PID_summary_key = tf.get_default_graph().unique_name(
            'PID_params_summary')

        if FLAGS.latent_u:
            U_goalie = p_U.goalie
            U_ball = p_U.ball
        else:
            U_goalie = p_G.goalie.u
            U_ball = p_G.ball.u

        Kp_goalie = tf.summary.scalar('PID_params/goalie/Kp',
                                      U_goalie.Kp[0, 0],
                                      collections=PID_summary_key)
        Ki_goalie = tf.summary.scalar('PID_params/goalie/Ki',
                                      U_goalie.Ki[0, 0],
                                      collections=PID_summary_key)
        Kd_goalie = tf.summary.scalar('PID_params/goalie/Kd',
                                      U_goalie.Kd[0, 0],
                                      collections=PID_summary_key)
        Kp_ball_x = tf.summary.scalar('PID_params/ball_x/Kp',
                                      U_ball.Kp[0, 0],
                                      collections=PID_summary_key)
        Ki_ball_x = tf.summary.scalar('PID_params/ball_x/Ki',
                                      U_ball.Ki[0, 0],
                                      collections=PID_summary_key)
        Kd_ball_x = tf.summary.scalar('PID_params/ball_x/Kd',
                                      U_ball.Kd[0, 0],
                                      collections=PID_summary_key)
        Kp_ball_y = tf.summary.scalar('PID_params/ball_y/Kp',
                                      U_ball.Kp[1, 0],
                                      collections=PID_summary_key)
        Ki_ball_y = tf.summary.scalar('PID_params/ball_y/Ki',
                                      U_ball.Ki[1, 0],
                                      collections=PID_summary_key)
        Kd_ball_y = tf.summary.scalar('PID_params/ball_y/Kd',
                                      U_ball.Kd[1, 0],
                                      collections=PID_summary_key)

        PID_summary = tf.summary.merge([Kp_goalie, Ki_goalie, Kd_goalie,
                                        Kp_ball_x, Ki_ball_x, Kd_ball_x,
                                        Kp_ball_y, Ki_ball_y, Kd_ball_y],
                                       collections=PID_summary_key,
                                       name='PID_params_summary')

    if FLAGS.save_posterior:
        with tf.name_scope('posterior'):
            with tf.name_scope('goal'):
                q_G_mean = tf.squeeze(q_G.postX, -1, name='mean')
                q_G_samp = tf.identity(q_G.sample(FLAGS.n_post_samples),
                                       name='samples')
            with tf.name_scope('control_signal'):
                if FLAGS.latent_u:
                    q_U_mean = tf.squeeze(q_U.postX, -1, name='mean')
                    q_U_samp = tf.identity(q_U.sample(FLAGS.n_post_samples),
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
    if FLAGS.model_type == 'VI_KLqp':
        n_batches = math.ceil(train_ntrials / FLAGS.B)
        if FLAGS.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(FLAGS.lr)

        if FLAGS.latent_u:
            var_list = (p_G.getParams() + p_U.getParams() +
                        q_G.getParams() + q_U.getParams())
            latent_vars = {p_G: q_G, p_U: q_U}
        else:
            var_list = (p_G.getParams() + q_G.getParams())
            latent_vars = {p_G: q_G}

        if FLAGS.profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            inference = KLqp_profile(options, run_metadata,
                                     latent_vars=latent_vars, data={Y: Y_ph})
        else:
            inference = ed.KLqp(latent_vars=latent_vars, data={Y: Y_ph})

        inference.initialize(n_iter=FLAGS.n_epochs * n_batches,
                             n_samples=FLAGS.n_samples,
                             scale={Y: 1 / FLAGS.B},
                             var_list=var_list,
                             optimizer=optimizer,
                             logdir=FLAGS.model_dir + '/log')

        sess = ed.get_session()
        tf.global_variables_initializer().run()

        lowest_ev_cost = np.Inf
        seso_saver = tf.train.Saver(tf.global_variables(),
                                    max_to_keep=FLAGS.max_ckpt_to_keep)
        lve_saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=2)

        if FLAGS.load_saved_model:
            seso_saver.restore(sess, FLAGS.saved_model_dir)
            print('Model restored from ' + FLAGS.saved_model_dir)

        # time2 = time.time()
        # print('Model setup took %.3f s.' % (time2 - time1))

        for i in range(FLAGS.n_epochs):
            if FLAGS.synthetic_data:
                batches = next(batch_generator(train_data, FLAGS.B))
            else:
                batches, conds, ctrls = next(batch_generator_pad(
                    train_data, FLAGS.B, train_conds, train_ctrls))

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

            if (i + 1) % FLAGS.frequency_saving_ckpt == 0:
                seso_saver.save(sess, FLAGS.model_dir + '/saved_model',
                                global_step=(i + 1),
                                latest_filename='checkpoint')

            if (i + 1) % FLAGS.frequency_val_loss == 0:
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
                      (i + 1, val_loss * FLAGS.B / val_ntrials))

                if val_loss * FLAGS.B / val_ntrials < lowest_ev_cost:
                    print('Saving model with lowest validation loss...')
                    lowest_ev_cost = val_loss * FLAGS.B / val_ntrials
                    lve_saver.save(sess, FLAGS.model_dir + '/saved_model_lve',
                                   global_step=(i + 1),
                                   latest_filename='checkpoint_lve',
                                   write_meta_graph=False)

        if FLAGS.profile:
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # use absolute path for FLAGS.model_dir
            with open(FLAGS.model_dir + '/timeline_01_step_%d.json' %
                      (i + 1), 'w') as f:
                f.write(chrome_trace)
                f.close()

        seso_saver.save(sess, FLAGS.model_dir + '/final_model')

        print('Model has been saved. Training ends.')

        # time3 = time.time()
        # print('Model training took %.3f s.' % (time3 - time2))


def main(_):
    if FLAGS.device_type == 'CPU':
        with tf.device('cpu:0'):
            run_model(FLAGS)
    elif FLAGS.device_type == 'GPU':
        run_model(FLAGS)
    else:
        raise Exception('Device type not recognized.')


if __name__ == '__main__':
    tf.app.run()

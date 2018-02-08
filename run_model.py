import os
import math
import tensorflow as tf
import numpy as np
import edward as ed
from tf_gbds.agents import game_model
from tf_gbds.utils import (load_data, get_max_velocities, get_vel, get_accel,
                           get_agent_params, generate_trial, pad_batch,
                           # batch_generator, batch_generator_pad,
                           KLqp_profile, add_summary)
import time
from tensorflow.python.client import timeline


# default flag values
MODEL_DIR = "new_model"
DATA_DIR = None
VAL_DIR = None
SYNTHETIC_DATA = False
SAVE_POSTERIOR = True
LOAD_SAVED_MODEL = False
SAVED_MODEL_DIR = None
PROFILE = False

N_AGENTS = 2
AGENT_NAME = "goalie,kicker"
AGENT_COLUMN = "0;1,2"
OBSERVE_DIM = 3
EXTRA_CONDITIONS = False
EXTRA_DIM = 0
OBSERVED_CONTROL = False
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

CONTROL_RESIDUAL_TOLERANCE = 1e-5
CONTROL_RESIDUAL_PENALTY = 1e8
CONTROL_ERROR_PENALTY = None
LATENT_CONTROL = False
CLIP = False
CLIP_RANGE_L = -1.
CLIP_RANGE_U = 1.
CLIP_TOLERANCE = 1e-5
CLIP_PENALTY = 1e8

SEED = 1234
TRAIN_RATIO = 0.85
OPTIMIZER = "Adam"
LEARNING_RATE = 1e-3
N_EPOCHS = 500
BATCH_SIZE = 1
N_VI_SAMPLES = 1
N_POSTERIOR_SAMPLES = 30
MAX_CKPT = 5
FREQ_CKPT = 5
FREQ_VAL_LOSS = 5


flags = tf.app.flags

flags.DEFINE_string("model_dir", MODEL_DIR,
                    "Directory where the model is saved")
flags.DEFINE_string("data_dir", DATA_DIR, "Directory of training data file")
flags.DEFINE_string("val_dir", VAL_DIR, "Directory of validation data file")
flags.DEFINE_boolean("synthetic_data", SYNTHETIC_DATA,
                     "Is the model trained on synthetic dataset")
flags.DEFINE_boolean("save_posterior", SAVE_POSTERIOR, "Will posterior \
                     samples be saved after training")
flags.DEFINE_boolean("load_saved_model", LOAD_SAVED_MODEL, "Is the model \
                     restored from an existing checkpoint")
flags.DEFINE_string("saved_model_dir", SAVED_MODEL_DIR,
                    "Directory where the model to be restored is saved")
flags.DEFINE_boolean("profile", PROFILE, "Is the model being profiled \
                     (use absolute path for FLAGS.model_dir if profiling)")
flags.DEFINE_string("device", "CPU",
                    "The device where the model is trained (CPU or GPU)")

flags.DEFINE_integer("n_agents", N_AGENTS, "Number of agents in the model")
flags.DEFINE_string("agent_name", AGENT_NAME, "Name of each agent \
                    (separated by ,)")
flags.DEFINE_string("agent_col", AGENT_COLUMN, "Columns of data \
                    corresponding to each agent (separated by ; and ,)")
flags.DEFINE_integer("obs_dim", OBSERVE_DIM, "Dimension of observation")
flags.DEFINE_boolean("extra_conds", EXTRA_CONDITIONS, "Are extra conditions \
                     included in the dataset")
flags.DEFINE_integer("extra_dim", EXTRA_DIM, "Dimension of extra conditions")
flags.DEFINE_boolean("ctrl_obs", OBSERVED_CONTROL, "Are observed control \
                     signals included in the dataset")
flags.DEFINE_boolean("add_accel", ADD_ACCEL,
                     "Is acceleration included in state")

flags.DEFINE_integer("GMM_K", GMM_K, "Number of components in GMM")
flags.DEFINE_integer("gen_n_layers", GEN_N_LAYERS, "Number of layers in \
                     neural networks (generative model)")
flags.DEFINE_integer("gen_hidden_dim", GEN_HIDDEN_DIM,
                     "Number of hidden units in each dense layer of \
                     neural networks (generative model)")
flags.DEFINE_integer("rec_lag", REC_LAG, "Number of previous timepoints \
                     included as input to recognition model")
flags.DEFINE_integer("rec_n_layers", REC_N_LAYERS, "Number of layers in \
                     neural networks (recognition model)")
flags.DEFINE_integer("rec_hidden_dim", REC_HIDDEN_DIM,
                     "Number of hidden units in each dense layer of \
                     neural networks (recognition model)")

flags.DEFINE_float("sigma", SIGMA, "Initial value of goal state variance")
flags.DEFINE_boolean("sigma_trainable", SIGMA_TRAINABLE, "Is sigma trainable")
flags.DEFINE_float("g_lb", GOAL_BOUNDARY_L, "Goal state lower boundary")
flags.DEFINE_float("g_ub", GOAL_BOUNDARY_U, "Goal state upper boundary")
flags.DEFINE_float("g_bounds_pen", GOAL_BOUNDARY_PENALTY,
                   "Penalty on goal states escaping boundaries")

flags.DEFINE_float("u_res_tol", CONTROL_RESIDUAL_TOLERANCE,
                   "Tolerance of control signal residual")
flags.DEFINE_float("u_res_pen", CONTROL_RESIDUAL_PENALTY,
                   "Penalty on control signal residual")
flags.DEFINE_float("u_error_pen", CONTROL_ERROR_PENALTY,
                   "Penalty on large control errors (input to PID control)")
flags.DEFINE_boolean("latent_u", LATENT_CONTROL, "Is the true control signal \
                     modeled as latent variable")
flags.DEFINE_boolean("clip", CLIP, "Is the observed control signal censored")
flags.DEFINE_float("clip_lb", CLIP_RANGE_L,
                   "Control signal censoring lower bound")
flags.DEFINE_float("clip_ub", CLIP_RANGE_U,
                   "Control signal censoring upper bound")
flags.DEFINE_float("clip_tol", CLIP_TOLERANCE,
                   "Tolerance of control signal censoring")
flags.DEFINE_float("clip_pen", CLIP_PENALTY,
                   "Penalty on control signal censoring")

flags.DEFINE_integer("seed", SEED, "Random seed for numpy functions")
flags.DEFINE_float("train_ratio", TRAIN_RATIO,
                   "Proportion of data used for training")
flags.DEFINE_string("opt", OPTIMIZER, "Gradient descent optimizer")
flags.DEFINE_float("lr", LEARNING_RATE, "Initial learning rate")
flags.DEFINE_integer("n_epochs", N_EPOCHS, "Number of iterations algorithm \
                    runs through the training set")
flags.DEFINE_integer("B", BATCH_SIZE, "Size of mini-batches")
flags.DEFINE_integer("n_samp", N_VI_SAMPLES, "Number of samples drawn \
                     for gradient estimation")
flags.DEFINE_integer("n_post_samp", N_POSTERIOR_SAMPLES, "Number of samples \
                     from posterior distributions to draw and save")
flags.DEFINE_integer("max_ckpt", MAX_CKPT,
                     "Maximum number of checkpoints to keep in the directory")
flags.DEFINE_integer("freq_ckpt", FREQ_CKPT, "Frequency of saving \
                     checkpoints to the directory")
flags.DEFINE_integer("freq_val_loss", FREQ_VAL_LOSS, "Frequency of \
                     evaluating loss of validation set")

FLAGS = flags.FLAGS


def run_model(FLAGS):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    agent_name = FLAGS.agent_name.split(",")
    assert len(agent_name) == FLAGS.n_agents, "The length of name list %i \
        does not match the number of agents %i" % (len(agent_name),
                                                   FLAGS.n_agents)

    agent_col = [[int(c) for c in a.split(",")]
                 for a in FLAGS.agent_col.split(";")]
    assert len(agent_col) == FLAGS.n_agents, "The length of column list %i \
        does not match the number of agents %i" % (len(agent_col),
                                                   FLAGS.n_agents)

    agent_dim = [len(a) for a in agent_col]
    assert sum(agent_dim) <= FLAGS.obs_dim, "The modeling dimensions exceed \
        observed: %i > %i" % (sum(agent_dim), FLAGS.obs_dim)

    with tf.name_scope("data"):
        with tf.name_scope("load_data"):
            train_set, val_set = load_data(FLAGS)
            print("Data loaded.")

        with tf.name_scope("get_max_velocities"):
            if val_set is None:
                all_vel = get_max_velocities([train_set], FLAGS.obs_dim)
            else:
                all_vel = get_max_velocities([train_set, val_set],
                                             FLAGS.obs_dim)
            print("The maximum velocity is %s." % all_vel)

        with tf.name_scope("get_iterator"):
            with tf.name_scope("training_set"):
                train_set = train_set.shuffle(buffer_size=10000)
                train_set = train_set.apply(
                    tf.contrib.data.batch_and_drop_remainder(FLAGS.B))
                # if FLAGS.B > 1:
                #     train_set = train_set.map(_pad_data)
                train_iterator = train_set.make_initializable_iterator(
                    "training_iterator")
                train_data = train_iterator.get_next("training_data")
            with tf.name_scope("validation_set"):
                if val_set is not None:
                    val_set = val_set.batch(1)
                    val_iterator = val_set.make_initializable_iterator(
                        "validation_iterator")
                    val_data = val_iterator.get_next("validation_data")
                else:
                    val_data = None

    if FLAGS.add_accel:
        state_dim = FLAGS.obs_dim * 3
        get_state = get_accel
    else:
        state_dim = FLAGS.obs_dim * 2
        get_state = get_vel

    if FLAGS.g_lb is not None and FLAGS.g_ub is not None:
        g_bounds = [FLAGS.g_lb, FLAGS.g_ub]
    else:
        g_bounds = None

    # No CLI arguments for these bc no longer being used,
    # but left just in case
    PKLparams = None
    penalty_Q = None

    if FLAGS.clip_lb is not None and FLAGS.clip_ub is not None:
        clip_range = [FLAGS.clip_lb, FLAGS.clip_ub]
    else:
        clip_range = None

    agent_params = []
    for i in range(FLAGS.n_agents):
        params = get_agent_params(
            agent_name[i], agent_dim[i], agent_col[i],
            FLAGS.obs_dim, state_dim, FLAGS.extra_dim,
            FLAGS.gen_n_layers, FLAGS.gen_hidden_dim,
            FLAGS.GMM_K, PKLparams, FLAGS.sigma, FLAGS.sigma_trainable,
            g_bounds, FLAGS.g_bounds_pen, all_vel, FLAGS.latent_u,
            FLAGS.rec_lag, FLAGS.rec_n_layers, FLAGS.rec_hidden_dim,
            penalty_Q, FLAGS.u_res_tol, FLAGS.u_res_pen,
            FLAGS.clip, clip_range, FLAGS.clip_tol, FLAGS.clip_pen,
            FLAGS.u_error_pen)
        agent_params.append(params)

    with tf.name_scope("inputs"):
        with tf.name_scope("trajectories_states"):
            inputs = {"trajectories": train_data["trajectory"],
                      "states": get_state(train_data["trajectory"], all_vel)}

        with tf.name_scope("extra_conditions"):
            if FLAGS.extra_conds:
                inputs.update({"extra_conds": train_data["extra_conds"]})
        
        with tf.name_scope("observed_control"):
            if FLAGS.ctrl_obs:
                inputs.update({"ctrl_obs": train_data["ctrl_obs"]})

    model = game_model(agent_params, inputs, FLAGS.n_post_samp,
                       name="penaltykick")

    # with tf.name_scope("generate_trial"):
    #     y0 = tf.placeholder(tf.float32, shape=(model.g.dim),
    #                         name="initial_position")
    #     max_trial_len = tf.placeholder(tf.int32, shape=(),
    #                                    name="maximum_trial_length")
    #     if FLAGS.extra_conds:
    #         gen_extra_conds = tf.placeholder(
    #             tf.float32, shape=FLAGS.extra_dim, name="extra_conditions")
    #     else:
    #         gen_extra_conds = None

    #     generated_trial, _, _ = generate_trial(
    #         model.g, model.u, y0, max_trial_len, gen_extra_conds)


    print("--------------Generative Parameters---------------")
    print("Number of GMM components: %i" % FLAGS.GMM_K)
    print("Number of layers in neural networks: %i" % FLAGS.gen_n_layers)
    print("Dimensions of hidden layers: %i" % FLAGS.gen_hidden_dim)
    if FLAGS.g_bounds_pen is not None:
        print("Penalty on goal states leaving boundary (Generative): %i"
              % FLAGS.g_bounds_pen)
    if FLAGS.u_error_pen is not None:
        print("Penalty on large control errors (Generative): %i"
              % FLAGS.u_error_pen)

    print("--------------Recognition Parameters--------------")
    print("Number of layers in neural networks: %i" % FLAGS.rec_n_layers)
    print("Dimensions of hidden layers: %i" % FLAGS.rec_hidden_dim)
    print("Lag of input: %i" % FLAGS.rec_lag)

    # with tf.name_scope("PID_params_summary"):
    #     PID_summary_key = tf.get_default_graph().unique_name(
    #         "PID_params_summary")

    #     if FLAGS.latent_u:
    #         U_goalie = p_U.goalie
    #         U_ball = p_U.ball
    #     else:
    #         U_goalie = p_G.goalie.u
    #         U_ball = p_G.ball.u

    #     Kp_goalie = tf.summary.scalar("PID_params/goalie/Kp",
    #                                   U_goalie.Kp[0, 0],
    #                                   collections=PID_summary_key)
    #     Ki_goalie = tf.summary.scalar("PID_params/goalie/Ki",
    #                                   U_goalie.Ki[0, 0],
    #                                   collections=PID_summary_key)
    #     Kd_goalie = tf.summary.scalar("PID_params/goalie/Kd",
    #                                   U_goalie.Kd[0, 0],
    #                                   collections=PID_summary_key)
    #     Kp_ball_x = tf.summary.scalar("PID_params/ball_x/Kp",
    #                                   U_ball.Kp[0, 0],
    #                                   collections=PID_summary_key)
    #     Ki_ball_x = tf.summary.scalar("PID_params/ball_x/Ki",
    #                                   U_ball.Ki[0, 0],
    #                                   collections=PID_summary_key)
    #     Kd_ball_x = tf.summary.scalar("PID_params/ball_x/Kd",
    #                                   U_ball.Kd[0, 0],
    #                                   collections=PID_summary_key)
    #     Kp_ball_y = tf.summary.scalar("PID_params/ball_y/Kp",
    #                                   U_ball.Kp[1, 0],
    #                                   collections=PID_summary_key)
    #     Ki_ball_y = tf.summary.scalar("PID_params/ball_y/Ki",
    #                                   U_ball.Ki[1, 0],
    #                                   collections=PID_summary_key)
    #     Kd_ball_y = tf.summary.scalar("PID_params/ball_y/Kd",
    #                                   U_ball.Kd[1, 0],
    #                                   collections=PID_summary_key)

    #     PID_summary = tf.summary.merge([Kp_goalie, Ki_goalie, Kd_goalie,
    #                                     Kp_ball_x, Ki_ball_x, Kd_ball_x,
    #                                     Kp_ball_y, Ki_ball_y, Kd_ball_y],
    #                                    collections=PID_summary_key,
    #                                    name="PID_params_summary")

    # if FLAGS.save_posterior:
    #     with tf.name_scope("posterior"):
    #         with tf.name_scope("goal"):
    #             q_G_mean = tf.squeeze(q_G.postX, -1, name="mean")
    #             q_G_samp = tf.identity(q_G.sample(FLAGS.n_post_samples),
    #                                    name="samples")
    #         with tf.name_scope("control_signal"):
    #             if FLAGS.latent_u:
    #                 q_U_mean = tf.squeeze(q_U.postX, -1, name="mean")
    #                 q_U_samp = tf.identity(q_U.sample(FLAGS.n_post_samples),
    #                                        name="samples")
    #         with tf.name_scope("GMM_goalie"):
    #             GMM_mu, GMM_lambda, GMM_w, _ = p_G.goalie.get_preds(
    #                 Y_ph, training=True,
    #                 post_g=tf.gather(q_G.sample(), p1_cols, axis=-1))
    #             g0_mu = tf.identity(p_G.goalie.g0_mu, name="g0_mu")
    #             g0_lambda = tf.identity(p_G.goalie.g0_lambda,
    #                                     name="g0_lambda")
    #             g0_w = tf.identity(p_G.goalie.g0_w, name="g0_w")
    #         with tf.name_scope("GMM_ball"):
    #             GMM_mu, GMM_lambda, GMM_w, _ = p_G.ball.get_preds(
    #                 Y_ph, training=True,
    #                 post_g=tf.gather(q_G.sample(), p2_cols, axis=-1))
    #             g0_mu = tf.identity(p_G.ball.g0_mu, name="g0_mu")
    #             g0_lambda = tf.identity(p_G.ball.g0_lambda, name="g0_lambda")
    #             g0_w = tf.identity(p_G.ball.g0_w, name="g0_w")

    # Variational inference (Edward KLqp)
    if FLAGS.profile:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        inference = KLqp_profile(
            options, run_metadata, model.latent_vars, model.data)
    else:
        inference = ed.KLqp(model.latent_vars, model.data)

    if FLAGS.opt == "Adam":
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    inference.initialize(n_samples=FLAGS.n_samp,
                         var_list=model.var_list,
                         optimizer=optimizer,
                         logdir=FLAGS.model_dir + "/log")

    if val_data is not None:
        val_inputs = {"trajectories": val_data["trajectory"],
                      "states": get_state(val_data["trajectory"], all_vel)}
        if FLAGS.extra_conds:
            val_inputs.update({"extra_conds": val_data["extra_conds"]})
        if FLAGS.ctrl_obs:
            val_inputs.update({"ctrl_obs": val_data["ctrl_obs"]})

        val_model = game_model(agent_params, val_inputs, FLAGS.n_post_samp,
                               name="validation")
        val_inference = ed.KLqp(val_model.latent_vars, val_model.data)
        val_inference.initialize(n_samples=FLAGS.n_samp,
                                 var_list=val_model.var_list)
        val_loss = tf.identity(val_inference.loss, name="validation_loss")

        # dict_swap = {train_data["trajectory"]: val_data["trajectory"]}
        # if FLAGS.extra_conds:
        #     dict_swap.update(
        #         {train_data["extra_conds"]: val_data["extra_conds"]})
        # if FLAGS.ctrl_obs:
        #     dict_swap.update({train_data["ctrl_obs"]: val_data["ctrl_obs"]})

        # val_loss = ed.copy(inference.loss, dict_swap, "validation_loss")

    print("Computational graph constructed.")

    sess = ed.get_session()
    tf.global_variables_initializer().run()

    seso_saver = tf.train.Saver(tf.global_variables(),
                                max_to_keep=FLAGS.max_ckpt,
                                name="session_saver")
    if val_data is not None:
        lve_saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=2,
                                   name="lowest_validation_loss_saver")
    lowest_ev_cost = np.Inf

    if FLAGS.load_saved_model:
        seso_saver.restore(sess, FLAGS.saved_model_dir)
        print("Model restored from " + FLAGS.saved_model_dir)

    for i in range(FLAGS.n_epochs):
        sess.run(train_iterator.initializer)

        while True:
            try:
                inference.update()
                # add_summary(PID_summary, inference, sess, feed_dict,
                #             info_dict["t"])
            except tf.errors.OutOfRangeError:
                break

        if (i + 1) % FLAGS.freq_ckpt == 0:
            seso_saver.save(sess, FLAGS.model_dir + "/saved_model",
                            global_step=(i + 1),
                            latest_filename="checkpoint")

        if val_data is not None:
            if (i + 1) % FLAGS.freq_val_loss == 0:
                sess.run(val_iterator.initializer)
                batch_loss = []

                while True:
                    try:
                        batch_loss.append(sess.run(val_loss))
                    except tf.errors.OutOfRangeError:
                        break

                print("\n", "Validation loss after epoch %i: %.3f" %
                      (i + 1, np.mean(batch_loss) / FLAGS.B))

                if (np.mean(batch_loss) / FLAGS.B) < lowest_ev_cost:
                    print("Saving model with lowest validation loss...")
                    lowest_ev_cost = np.mean(batch_loss) / FLAGS.B
                    lve_saver.save(sess, FLAGS.model_dir + "/saved_model_lve",
                                   global_step=(i + 1),
                                   latest_filename="checkpoint_lve",
                                   write_meta_graph=False)

    if FLAGS.profile:
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # use absolute path for FLAGS.model_dir
        with open(FLAGS.model_dir + "/timeline_01_step_%d.json" %
                  (i + 1), "w") as f:
            f.write(chrome_trace)
            f.close()

    seso_saver.save(sess, FLAGS.model_dir + "/final_model")
    sess.close()

    print("Training completed.")


def main(_):
    if FLAGS.device == "CPU":
        with tf.device('/cpu:0'):
            run_model(FLAGS)
    elif FLAGS.device == "GPU":
        run_model(FLAGS)
    else:
        raise Exception("Device type not recognized.")


if __name__ == "__main__":
    tf.app.run()

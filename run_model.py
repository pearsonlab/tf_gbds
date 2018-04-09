import os
import tensorflow as tf
# import numpy as np
import edward as ed
from tf_gbds.agents import game_model
from tf_gbds.utils import (load_data, get_max_velocities, get_vel, get_accel,
                           get_model_params, pad_batch, add_summary,
                           KLqp_profile, KLqp_grad_clipnorm)
import time
from tensorflow.python.client import timeline


# default flag values
MODEL_DIR = "new_model"
DATA_DIR = None
SYNTHETIC_DATA = False
SAVE_POSTERIOR = True
LOAD_SAVED_MODEL = False
SAVED_MODEL_DIR = None
PROFILE = False

GAME_NAME = "penaltykick"
N_AGENTS = 2
AGENT_NAME = "goalie,shooter"
AGENT_COLUMN = "0;1,2"
OBSERVE_DIM = 3
OBSERVED_CONTROL = False
EXTRA_CONDITIONS = False
EXTRA_DIM = 0
ADD_ACCEL = False

GMM_K = 8
GEN_N_LAYERS = 3
GEN_HIDDEN_DIM = 64
REC_LAG = 10
REC_N_LAYERS = 3
REC_HIDDEN_DIM = 64

SIGMA = 1e-3
SIGMA_TRAINABLE = False
SIGMA_PENALTY = 1e3
GOAL_BOUNDARY_L = -1.
GOAL_BOUNDARY_U = 1.
GOAL_BOUNDARY_PENALTY = None

# CONTROL_RESIDUAL_TOLERANCE = 1e-5
# CONTROL_RESIDUAL_PENALTY = 1e8
EPSILON = 1e-5
EPSILON_TRAINABLE = False
EPSILON_PENALTY = 1e5
CONTROL_ERROR_TOLERANCE = .5
CONTROL_ERROR_PENALTY = None
LATENT_CONTROL = False
CLIP = False
CLIP_RANGE_L = -1.
CLIP_RANGE_U = 1.
CLIP_TOLERANCE = 1e-5
CLIP_PENALTY = 1e8

OPTIMIZER = "Adam"
LEARNING_RATE = 1e-3
N_EPOCHS = 500
BATCH_SIZE = 1
N_VI_SAMPLES = 1
N_POSTERIOR_SAMPLES = 30
MAX_CKPT = 3
FREQ_CKPT = 5


flags = tf.app.flags

flags.DEFINE_string("model_dir", MODEL_DIR,
                    "Directory where the model is saved")
flags.DEFINE_string("data_dir", DATA_DIR, "Directory of training data file")
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
# flags.DEFINE_string("device", "CPU",
#                     "The device where the model is trained (CPU or GPU)")

flags.DEFINE_string("game_name", GAME_NAME, "Name of the game")
flags.DEFINE_integer("n_agents", N_AGENTS, "Number of agents in the model")
flags.DEFINE_string("agent_name", AGENT_NAME, "Name of each agent \
                    (separated by ,)")
flags.DEFINE_string("agent_col", AGENT_COLUMN, "Columns of data \
                    corresponding to each agent (separated by ; and ,)")
flags.DEFINE_integer("obs_dim", OBSERVE_DIM, "Dimension of observation")
flags.DEFINE_boolean("ctrl_obs", OBSERVED_CONTROL, "Are observed control \
                     signals included in the dataset")
flags.DEFINE_boolean("extra_conds", EXTRA_CONDITIONS, "Are extra conditions \
                     included in the dataset")
flags.DEFINE_integer("extra_dim", EXTRA_DIM, "Dimension of extra conditions")
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

flags.DEFINE_float("sigma", SIGMA, "(Initial) value of goal state variance")
flags.DEFINE_boolean("sigma_trainable", SIGMA_TRAINABLE, "Is sigma trainable")
flags.DEFINE_float("sigma_pen", SIGMA_PENALTY, "Penalty on large sigma")
flags.DEFINE_float("g_lb", GOAL_BOUNDARY_L, "Goal state lower boundary")
flags.DEFINE_float("g_ub", GOAL_BOUNDARY_U, "Goal state upper boundary")
flags.DEFINE_float("g_bounds_pen", GOAL_BOUNDARY_PENALTY,
                   "Penalty on goal states escaping boundaries")

# flags.DEFINE_float("u_res_tol", CONTROL_RESIDUAL_TOLERANCE,
#                    "Tolerance of control signal residual")
# flags.DEFINE_float("u_res_pen", CONTROL_RESIDUAL_PENALTY,
#                    "Penalty on control signal residual")
flags.DEFINE_float("eps", EPSILON, "(Initial) value of control signal noise")
flags.DEFINE_boolean("eps_trainable", EPSILON_TRAINABLE,
                     "Is epsilon trainable")
flags.DEFINE_float("eps_pen", EPSILON_PENALTY, "Penalty on large epsilon")
flags.DEFINE_float("u_error_tol", CONTROL_ERROR_TOLERANCE,
                   "Tolerance of control error (input to PID control model)")
flags.DEFINE_float("u_error_pen", CONTROL_ERROR_PENALTY,
                   "Penalty on large control errors")
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

FLAGS = flags.FLAGS


def run_model(FLAGS):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    # Check provided agent information
    agent_name = FLAGS.agent_name.split(",")
    assert len(agent_name) == FLAGS.n_agents, "The length of name list \
        %i does not match the number of agents %i" % (len(agent_name),
                                                      FLAGS.n_agents)

    agent_col = [[int(c) for c in a.split(",")]
                 for a in FLAGS.agent_col.split(";")]
    assert len(agent_col) == FLAGS.n_agents, "The length of column list \
        %i does not match the number of agents %i" % (len(agent_col),
                                                      FLAGS.n_agents)

    agent_dim = [len(a) for a in agent_col]
    assert sum(agent_dim) <= FLAGS.obs_dim, "The modeling dimension \
        exceeds observed: %i > %i" % (sum(agent_dim), FLAGS.obs_dim)

    agents = []
    for i in range(FLAGS.n_agents):
        agents.append(
            dict(name=agent_name[i], col=agent_col[i], dim=agent_dim[i]))

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

    # epoch = tf.placeholder(tf.int32, name="epoch")
    # with tf.name_scope("penalty"):
    #     u_res_pen_t = tf.multiply(
    #         FLAGS.u_res_pen, tf.to_float(10 ** tf.minimum(epoch // 20, 3)),
    #         "control_residual_penalty")
    #     u_res_tol_t = tf.divide(
    #         FLAGS.u_res_tol, tf.to_float(10 ** tf.minimum(epoch // 10, 5)),
    #         "control_residual_tolerance")

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

    with tf.device("/cpu:0"):
        with tf.name_scope("data"):
            with tf.name_scope("load_data"):
                train_set = load_data(FLAGS)
                print("Data loaded.")

            with tf.name_scope("get_max_velocities"):
                max_vel = get_max_velocities([train_set], FLAGS.obs_dim)
                print("The maximum velocity is %s." % max_vel)

            with tf.name_scope("get_iterator"):
                with tf.name_scope("training_set"):
                    train_set = train_set.shuffle(buffer_size=100000)
                    train_set = train_set.apply(
                        tf.contrib.data.batch_and_drop_remainder(FLAGS.B))
                    # if FLAGS.B > 1:
                    #     train_set = train_set.map(_pad_data)
                    train_iterator = train_set.make_initializable_iterator(
                        "training_iterator")
                    train_data = train_iterator.get_next("training_data")

        params = get_model_params(
            FLAGS.game_name, agents, FLAGS.obs_dim, state_dim,
            FLAGS.extra_dim, FLAGS.gen_n_layers, FLAGS.gen_hidden_dim,
            FLAGS.GMM_K, PKLparams,
            FLAGS.sigma, FLAGS.sigma_trainable, FLAGS.sigma_pen,
            g_bounds, FLAGS.g_bounds_pen, FLAGS.latent_u,
            FLAGS.rec_lag, FLAGS.rec_n_layers, FLAGS.rec_hidden_dim,
            penalty_Q, FLAGS.eps, FLAGS.eps_trainable, FLAGS.eps_pen,
            FLAGS.u_error_tol, FLAGS.u_error_pen,
            FLAGS.clip, clip_range, FLAGS.clip_tol, FLAGS.clip_pen)

        with tf.name_scope("inputs"):
            y_train = tf.identity(train_data["trajectory"], "trajectory")
            s_train = tf.identity(get_state(y_train, max_vel), "states")
            if FLAGS.ctrl_obs:
                ctrl_obs_train = tf.identity(train_data["ctrl_obs"],
                                             "observed_control")
            else:
                with tf.name_scope("observed_control"):
                    ctrl_obs_train = tf.atanh(tf.divide(tf.subtract(
                        y_train[:, 1:], y_train[:, :-1], "diff"),
                        max_vel + 1e-8, "standardize"), "arctanh")
            if FLAGS.extra_conds:
                extra_conds_train = tf.identity(train_data["extra_conds"],
                                                "extra_conditions")
            else:
                extra_conds_train = None

            inputs = {"trajectory": y_train, "states": s_train,
                      "ctrl_obs": ctrl_obs_train,
                      "extra_conds": extra_conds_train}

        model = game_model(params, inputs, max_vel, FLAGS.extra_dim,
                           FLAGS.n_post_samp)

        with tf.name_scope("PID_params_summary"):
            PID_summary_key = tf.get_default_graph().unique_name(
                "PID_params_summary")

            Kp_goalie_y = tf.summary.scalar("PID_params/goalie_y/Kp",
                                            model.p.agents[0].Kp[0],
                                            collections=PID_summary_key)
            Ki_goalie_y = tf.summary.scalar("PID_params/goalie_y/Ki",
                                            model.p.agents[0].Ki[0],
                                            collections=PID_summary_key)
            Kd_goalie_y = tf.summary.scalar("PID_params/goalie_y/Kd",
                                            model.p.agents[0].Kd[0],
                                            collections=PID_summary_key)
            Kp_shooter_x = tf.summary.scalar("PID_params/shooter_x/Kp",
                                             model.p.agents[1].Kp[0],
                                             collections=PID_summary_key)
            Ki_shooter_x = tf.summary.scalar("PID_params/shooter_x/Ki",
                                             model.p.agents[1].Ki[0],
                                             collections=PID_summary_key)
            Kd_shooter_x = tf.summary.scalar("PID_params/shooter_x/Kd",
                                             model.p.agents[1].Kd[0],
                                             collections=PID_summary_key)
            Kp_shooter_y = tf.summary.scalar("PID_params/shooter_y/Kp",
                                             model.p.agents[1].Kp[1],
                                             collections=PID_summary_key)
            Ki_shooter_y = tf.summary.scalar("PID_params/shooter_y/Ki",
                                             model.p.agents[1].Ki[1],
                                             collections=PID_summary_key)
            Kd_shooter_y = tf.summary.scalar("PID_params/shooter_y/Kd",
                                             model.p.agents[1].Kd[1],
                                             collections=PID_summary_key)
            eps = tf.summary.scalar(
                "PID_params/eps", model.p.agents[0].eps[0, 0],
                collections=PID_summary_key)

            PID_summary = tf.summary.merge(
                [Kp_goalie_y, Ki_goalie_y, Kd_goalie_y,
                 Kp_shooter_x, Ki_shooter_x, Kd_shooter_x,
                 Kp_shooter_y, Ki_shooter_y, Kd_shooter_y, eps],
                collections=PID_summary_key, name="PID_params_summary")

        # Variational Inference (Edward KLqp)
        if FLAGS.profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            inference = KLqp_profile(options, run_metadata, model.latent_vars)
        else:
            # inference = ed.KLqp(model.latent_vars)
            inference = KLqp_grad_clipnorm(latent_vars=model.latent_vars)

        if FLAGS.opt == "Adam":
            optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        inference.initialize(n_samples=FLAGS.n_samp,
                             var_list=model.var_list,
                             optimizer=optimizer,
                             logdir=FLAGS.model_dir + "/log")

    print("Computational graph constructed.")

    sess = ed.get_session()
    tf.global_variables_initializer().run()

    seso_saver = tf.train.Saver(tf.global_variables(),
                                max_to_keep=FLAGS.max_ckpt,
                                name="session_saver")

    if FLAGS.load_saved_model:
        seso_saver.restore(sess, FLAGS.saved_model_dir)
        print("Parameters saved in %s restored." % FLAGS.saved_model_dir)

    for i in range(FLAGS.n_epochs):
        if i == 0 or (i + 1) % 5 == 0:
            print("Entering epoch %i ..." % (i + 1))

        train_iterator.initializer.run()

        while True:
            try:
                feed_dict = None
                info_dict = inference.update(feed_dict=feed_dict)
                add_summary(PID_summary, inference, sess, feed_dict,
                            info_dict["t"])
            except tf.errors.OutOfRangeError:
                break

        if (i + 1) % FLAGS.freq_ckpt == 0:
            seso_saver.save(sess, FLAGS.model_dir + "/saved_model",
                            global_step=(i + 1),
                            latest_filename="checkpoint")
            print("Model saved after epoch %i." % (i + 1))

    if FLAGS.profile:
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # use absolute path for FLAGS.model_dir
        with open(FLAGS.model_dir + "/timeline_01_step_%d.json" %
                  (i + 1), "w") as f:
            f.write(chrome_trace)
            f.close()

    seso_saver.save(sess, FLAGS.model_dir + "/final_model")
    inference.finalize()
    sess.close()

    print("Training completed.")


def main(_):
    run_model(FLAGS)


if __name__ == "__main__":
    tf.app.run()

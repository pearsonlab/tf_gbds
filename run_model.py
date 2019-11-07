import os
import sys
from os.path import join
# Directory setup and add subfolders
curr_dir = os.getcwd()
sys.path.append(join(curr_dir, 'utils'))

import numpy as np
import tensorflow as tf
import edward as ed
# import time
from tensorflow.python.client import timeline

# Load utilities
# from tf_util import *
from agents import game_model
from utils import (load_data, get_vel, get_accel, get_model_params,
                   add_summary, KLqp_profile)


# default flag values
MODEL_DIR = "model_1"
TRAINING_DATA_DIR = ""
VALIDATION_DATA_DIR = ""
SYNTHETIC_DATA = False
SAVE_POSTERIOR = True
LOAD_SAVED_MODEL = False
SAVED_MODEL_DIR = None
PROFILE = False

GAME_NAME = "pacman"
AGENT_NAME = "subject"
OBSERVE_DIM = 2
NPCS = True
N_NPCS = 2
MAX_VEL = ".021,.021"

GMM_K = 20
GEN_LAG = 2
GEN_N_LAYERS = 3
GEN_HIDDEN_DIM = 64
REC_LAG = 10
REC_N_LAYERS = 3
REC_HIDDEN_DIM = 64
LSTM_UNITS = 64

GOAL_BOUNDARY_L = -1.1
GOAL_BOUNDARY_U = 1.1
GOAL_BOUNDARY_PENALTY = 1e6
GOAL_PRECISION_PENALTY = 1.

EPSILON = -7.
EPSILON_TRAINABLE = False
EPSILON_PENALTY = 1e3

SEED = 1234
OPTIMIZER = "Adam"
LEARNING_RATE = 1e-3
N_EPOCHS = 500
BATCH_SIZE = 1
N_VI_SAMPLES = 1
N_POSTERIOR_SAMPLES = 30
MAX_CKPT = 10
FREQ_CKPT = 5
FREQ_VAL_LOSS = 1


flags = tf.app.flags

flags.DEFINE_string("model_dir", MODEL_DIR,
                    "Directory where the model is saved")
flags.DEFINE_string("train_data_dir", TRAINING_DATA_DIR,
                    "Directory of training data file")
flags.DEFINE_string("val_data_dir", VALIDATION_DATA_DIR,
                    "Directory of validation data file")
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

flags.DEFINE_string("game_name", GAME_NAME, "Name of the game")
flags.DEFINE_string("agent_name", AGENT_NAME, "Name of each agent \
                    (separated by ,)")
flags.DEFINE_integer("obs_dim", OBSERVE_DIM, "Dimension of observation")
flags.DEFINE_boolean("npcs", NPCS, "Are npcs information included in the \
                     dataset")
flags.DEFINE_integer("n_npcs", N_NPCS, "Number of npcs")
flags.DEFINE_string("max_vel", MAX_VEL, "Maximum velocity of agent")

flags.DEFINE_integer("GMM_K", GMM_K, "Number of components in GMM")
flags.DEFINE_integer("gen_lag", GEN_LAG, "Number of previous timepoints \
                     included as input to generate model")
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
flags.DEFINE_integer("lstm_units", LSTM_UNITS,
                     "Number of units in LSTM layer of recognition model")

flags.DEFINE_float("g_lb", GOAL_BOUNDARY_L, "Goal state lower boundary")
flags.DEFINE_float("g_ub", GOAL_BOUNDARY_U, "Goal state upper boundary")
flags.DEFINE_float("g_bounds_pen", GOAL_BOUNDARY_PENALTY,
                   "Penalty on goal states escaping boundaries")
flags.DEFINE_float("g_prec_pen", GOAL_PRECISION_PENALTY, "Penalty on \
                   the (unconstrained) precision of GMM's components")

flags.DEFINE_float("eps_init", EPSILON, "(Initial) unconstrained value of \
                   (latent if modeled) control signal standard deviation")
flags.DEFINE_boolean("eps_trainable", EPSILON_TRAINABLE,
                     "Is epsilon trainable")
flags.DEFINE_float("eps_pen", EPSILON_PENALTY, "Penalty on large epsilon")

flags.DEFINE_integer("seed", SEED, "Random seed")
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
flags.DEFINE_integer("freq_val_loss", FREQ_VAL_LOSS, "Frequency of computing \
                     validation set loss")

FLAGS = flags.FLAGS

ed.set_seed(FLAGS.seed)


def run_model(FLAGS):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    max_vel = np.array([float(n) for n in FLAGS.max_vel.split(",")],
                       dtype=np.float32)

    if FLAGS.g_lb is not None and FLAGS.g_ub is not None:
        g_bounds = [FLAGS.g_lb, FLAGS.g_ub]
    else:
        g_bounds = None

    print("--------------Generative Parameters---------------")
    print("Number of GMM components: %i" % FLAGS.GMM_K)
    print("Lag of input: %i" % FLAGS.gen_lag)
    print("Number of layers in neural networks: %i" % FLAGS.gen_n_layers)
    print("Dimensions of hidden layers: %i" % FLAGS.gen_hidden_dim)
    if FLAGS.g_bounds_pen is not None:
        print("Penalty on goal states leaving boundary (Generative): %i"
              % FLAGS.g_bounds_pen)
    # if FLAGS.g_prec_pen is not None:
    #     print("Penalty on the precision of GMM components (Generative) : %i"
    #           % FLAGS.g_prec_pen)

    print("--------------Recognition Parameters--------------")
    print("Lag of input: %i" % FLAGS.rec_lag)
    print("Number of layers in neural networks: %i" % FLAGS.rec_n_layers)
    print("Dimensions of hidden layers: %i" % FLAGS.rec_hidden_dim)
    print("Number of units in LSTM layer: %i" % FLAGS.lstm_units)

    with tf.device("/cpu:0"):
        init_temperature = 1.
        epoch = tf.placeholder(tf.int64, name="epoch")
        data_dir = tf.placeholder(tf.string, name="dataset_directory")

        with tf.name_scope("load_data"):
            iterator = load_data(data_dir, FLAGS)
            data = iterator.get_next("data")

            if FLAGS.npcs:
                (subject, npcs) = data
            else:
                (subject,) = data

            subject_in = tf.identity(subject, "subject")
            traj_in = tf.identity(
                subject[..., :FLAGS.obs_dim], "trajectory")

            if FLAGS.npcs:
                npcs_in = tf.identity(npcs, "npcs")
            else:
                npcs_in = None

            # with tf.name_scope("observed_control"):
            #     ctrl_obs_in = tf.divide(
            #         tf.subtract(traj_in[:, 1:], traj_in[:, :-1], "diff"),
            #         max_vel, "standardize")
            ctrl_obs_in = tf.identity(
                subject[..., FLAGS.obs_dim:], "observed_control")

            inputs = {"trajectory": traj_in, "npcs": npcs_in,
                      "observed_control": ctrl_obs_in}

        params = get_model_params(
            FLAGS.game_name, FLAGS.agent_name, FLAGS.obs_dim, FLAGS.n_npcs,
            FLAGS.gen_lag, FLAGS.GMM_K,
            FLAGS.gen_n_layers, FLAGS.gen_hidden_dim,
            g_bounds, FLAGS.g_bounds_pen, FLAGS.g_prec_pen,
            FLAGS.rec_lag, FLAGS.lstm_units,
            FLAGS.rec_n_layers, FLAGS.rec_hidden_dim,
            FLAGS.eps_init, FLAGS.eps_trainable, FLAGS.eps_pen,
            init_temperature, epoch)

        model = game_model(params, inputs, max_vel, FLAGS.n_post_samp)

        with tf.name_scope("parameters_summary"):
            summary_list = []

            Kp_x = tf.summary.scalar("Kp/x", model.p.Kp[0])
            Kp_y = tf.summary.scalar("Kp/y", model.p.Kp[1])
            summary_list += [Kp_x, Kp_y]

            if FLAGS.eps_trainable:
                eps_subject_x = tf.summary.scalar(
                    "epsilon/x", model.p.eps[0, 0])
                eps_subject_y = tf.summary.scalar(
                    "epsilon/y", model.p.eps[0, 1])
                # eps_pen = tf.summary.scalar(
                #     "epsilon/penalty", model.p.eps_pen)
                summary_list += [eps_subject_x, eps_subject_y]
                # summary_list += [eps_subject_x, eps_subject_y, eps_pen]

            # summary_list += [tf.summary.scalar(
            #     "temperature", model.p.temperature)]

            all_summary = tf.summary.merge(summary_list)

        # Variational Inference (Edward KLqp)
        if FLAGS.profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            inference = KLqp_profile(options, run_metadata, model.latent_vars)
        else:
            inference = ed.KLqp(model.latent_vars)
            # inference = KLqp_clipgrads(latent_vars=model.latent_vars)

        if FLAGS.opt == "Adam":
            optimizer = tf.train.AdamOptimizer(FLAGS.lr)

        inference.initialize(n_samples=FLAGS.n_samp,
                             var_list=model.var_list,
                             optimizer=optimizer,
                             logdir=FLAGS.model_dir + "/log",
                             log_vars=model.log_vars)

    print("Computational graph constructed.")

    sess = ed.get_session()
    tf.global_variables_initializer().run()

    sess_saver = tf.train.Saver(tf.global_variables(),
                                max_to_keep=FLAGS.max_ckpt,
                                name="session_saver")

    if FLAGS.load_saved_model:
        sess_saver.restore(sess, FLAGS.saved_model_dir)
        print("Parameters saved in %s restored." % FLAGS.saved_model_dir)

    val_loss = []

    print("Training initiated.")

    for i in range(FLAGS.n_epochs):
        if i == 0 or (i + 1) % 5 == 0:
            print("Entering epoch %s ..." % (i + 1))

        iterator.initializer.run({data_dir: FLAGS.train_data_dir})
        while True:
            try:
                feed_dict = {epoch: (i + 1)}
                info_dict = inference.update(feed_dict=feed_dict)
                add_summary(all_summary, inference, sess, feed_dict,
                            info_dict["t"])
            except tf.errors.OutOfRangeError:
                break

        if (i + 1) % FLAGS.freq_ckpt == 0:
            sess_saver.save(sess, FLAGS.model_dir + "/saved_model",
                            global_step=(i + 1), latest_filename="ckpt")
            print("Model saved after %s epochs." % (i + 1))

        if (i + 1) % FLAGS.freq_val_loss == 0:
            curr_val_loss = []
            iterator.initializer.run({data_dir: FLAGS.val_data_dir})
            while True:
                try:
                    curr_val_loss.append(sess.run(
                        inference.loss, {epoch: (i + 1)}))
                except tf.errors.OutOfRangeError:
                    break

            val_loss.append(np.array(curr_val_loss).mean())
            print("Validation set loss after epoch %s is %.3f." % (
                (i + 1), val_loss[-1]))
            np.save(FLAGS.model_dir + "/val_loss", val_loss)

    # if FLAGS.profile:
    #     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    #     chrome_trace = fetched_timeline.generate_chrome_trace_format()
    #     # use absolute path for FLAGS.model_dir
    #     with open(FLAGS.model_dir + "/timeline_01_step_%d.json" %
    #               (i + 1), "w") as f:
    #         f.write(chrome_trace)
    #         f.close()

    # seso_saver.save(sess, FLAGS.model_dir + "/final_model")
    inference.finalize()
    sess.close()

    print("Training completed.")


def main(_):
    run_model(FLAGS)


if __name__ == "__main__":
    tf.app.run()

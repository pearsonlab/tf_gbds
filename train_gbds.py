"""
Train VB and CGAN portions of model. CGAN can optionally be trained at a larger
scale on a GPU in train_CGAN_parallel.py
"""
import argparse
from utils import *
from prep_GAN import prep_GAN_data, prep_injection_conditions
import sys
import os
from os.path import expanduser
# sys.path.append(expanduser('~/code/gbds/code/'))
# sys.path.append(expanduser('~/code/gbds/code/lib/'))
from PenaltyKick import SGVB_GBDS
import tensorflow as tf
import numpy as np
import _pickle as pkl

# COMPILE_MODE = 'FAST_RUN'  # change this for debugging


def run_model(**kwargs):
    outname = kwargs['outname']
    seed = kwargs['seed']
    max_sessions = kwargs['max_sessions']
    session_type = kwargs['session_type']
    session_index = kwargs['session_index']
    data_loc = kwargs['data_loc']
    rec_lag = kwargs['rec_lag']
    nlayers_rec = kwargs['nlayers_rec']
    hidden_dim_rec = kwargs['hidden_dim_rec']
    add_accel = kwargs['add_accel']
    penalty_eps = kwargs['penalty_eps']
    penalty_sigma = kwargs['penalty_sigma']
    # model class allows for having 2 boundaries with different penalties,
    # but we later found that unnecessary, so the CLI only takes on penalty.
    # We left the possibility for 2 penalties in the model class just in case
    # it may be useful on a different dataset/task
    boundaries_g = (kwargs['boundary_g'], 2.0)
    penalty_g = (kwargs['penalty_g'], None)
    learning_rate = tf.cast(kwargs['learning_rate'], tf.float32)
    n_epochs = kwargs['n_epochs']

    np.random.seed(seed)  # set seed for consistent train/val split

    if not os.path.exists(outname):
        os.makedirs(outname)
    if session_type == 'recording':
        print "Loading movement data from recording sessions..."
        groups = get_session_names(session_index,
                                   ('type',), ('recording',))
    elif session_type == 'injection':
        print "Loading movement data from injection sessions..."
        groups = get_session_names(session_index,
                                   ('type', 'type'),
                                   ('saline', 'muscimol'),
                                   comb=np.any)
    sys.stdout.flush()
    groups = groups[-max_sessions:]
    y_data, y_data_modes, y_val_data, y_val_data_modes = load_pk_data(data_loc,
                                                                      groups)

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

    vel = get_max_velocities(y_data, y_val_data)
    ntrials = len(y_data)

    rec_params = get_rec_params_GBDS(seed, obs_dim, rec_lag, nlayers_rec,
                                     hidden_dim_rec, penalty_Q,
                                     PKLparams)
    gen_params_g = get_gen_params_GBDS(seed, obs_dim_g, obs_dim, add_accel,
                                       yCols_goalie, nlayers_rec,
                                       hidden_dim_rec, PKLparams, vel,
                                       penalty_eps, penalty_sigma,
                                       boundaries_g, penalty_g)
    gen_params_b = get_gen_params_GBDS(seed, obs_dim_b, obs_dim, add_accel,
                                       yCols_ball, nlayers_rec, hidden_dim_rec,
                                       PKLparams, vel, penalty_eps,
                                       penalty_sigma, boundaries_g, penalty_g)

    model = SGVB_GBDS(gen_params_b, gen_params_g, yCols_ball, yCols_goalie,
                      rec_params, ntrials)

    print "Check params:"
    if add_pklayers:
        if row_sparse:
            print "a: %f" % model.all_PKbias_layers[0].a
            print "b: %f" % model.all_PKbias_layers[0].b
        else:
            print "k: %f" % model.all_PKbias_layers[0].k
    print "--------------Generative Params----------------"
    if penalty_eps is not None:
        print "Penalty on control signal noise, epsilon (Generative): %i" % penalty_eps
    if penalty_sigma is not None:
        print "Penalty on goal state noise, sigma (Generative): %i" % penalty_sigma
    if penalty_g[0] is not None:
        print "Penalty on goal state leaving boundary 1 (Generative): %i" % penalty_g[0]
    if penalty_g[1] is not None:
        print "Penalty on goal state leaving boundary 2 (Generative): %i" % penalty_g[1]
    print "--------------Recognition Params---------------"
    print "Num Layers (VILDS recognition): %i" % nlayers_rec
    print "Hidden Dims (VILDS recognition): %i" % hidden_dim_rec
    print "Input lag (VILDS recognition): %i" % rec_lag
    sys.stdout.flush()

    # set up an iterator over our training data
    data_iter_vb = MultiDatasetTrialIndexIterator((y_data, y_data_modes),
                                                  randomize=True)

    val_costs = []
    ctrl_cost = []

    # Iterate over the training data for the specified number of epochs
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        print "Compiling graph for VB model..."
        sys.stdout.flush()
        # ctrl_train_fn, ctrl_test_fn = (
        #     compile_functions_vb_training(model, learning_rate,
        #                                   add_pklayers=add_pklayers,
        #                                   mode=COMPILE_MODE))
        if add_pklayers:
            model.mode = tf.placeholder(tf.float32, name='batch_mode')
        if joint_spikes and model.isTrainingSpikeModel:
            model.spikes = tf.placeholder(tf.float32, name='batch_spikes')
            model.signals = tf.placeholder(tf.float32, name='batch_signals')

        var_list = model.getParams()
        for param in var_list:
        if param.name == 'W' or param.name == 'b' or param.name == 'G':
        # only on NN parameters
            param = tf.clip_by_norm(param, 5, axes=[0])
        if cap_noise and (param.name == 'QinvChol' or
                          param.name == 'Q0invChol'):
            param = tf.clip_by_norm(param, 30, axes=[0])

        opt = tf.train.AdamOptimizer(learning_rate)
        train_op = opt.minimize(-model.cost(), var_list=var_list)

        print '---> Training control model'
        sys.stdout.flush()
        for ie in np.arange(n_epochs):
            print('--> entering epoch %i' % (ie + 1))
            sys.stdout.flush()
            for data_pack in data_iter_vb:
                if add_pklayers:
                    y, mode = data_pack
                    _, cost = sess.run([train_op, model.cost()],
                                       feed_dict={model.Y:y, model.mode:mode})
                    ctrl_cost.append(cost)
                else:
                    y, mode = data_pack
                    _, cost = sess.run([train_op, model.cost()],
                                       feed_dict={model.Y:y})
                    ctrl_cost.append(cost)
            curr_val_costs = []
            for i in range(len(y_val_data)):
                if add_pklayers:
                    val_cost = sess.run(
                        model.cost(),
                        feed_dict={model.Y:y_val_data[i],
                                   model.mode:y_val_data_modes[i]})
                    curr_val_costs.append(val_cost)
                else:
                    val_cost = sess.run(
                        model.cost(),
                        feed_dict={model.Y:y_val_data[i]})
                    curr_val_costs.append(val_cost)
            val_costs.append(np.array(curr_val_costs).mean())
            print "Validation set cost: %f" % val_costs[-1]
            sys.stdout.flush()
            with open(outname + '/model.pkl', 'wb') as f:
                pkl.dump(model, f)
            np.save(outname + '/behav_train_costs', ctrl_cost)
            np.save(outname + '/behav_val_costs', val_costs)

        # save outputs to use during CGAN and GAN training
        print "----> Preparing GAN data"
        sys.stdout.flush()
        # add injection location and type as conditions to cGAN
        if session_type == 'injection':
            inject_conds = prep_injection_conditions(y_data, y_data_modes)
        else:
            inject_conds = None
        y_data_postJ, y_data_post_g0, y_data_states = prep_GAN_data(
            model, y_data, extra_cond=inject_conds)
        np.save(outname + '/postJ', y_data_postJ)
        np.save(outname + '/states', y_data_states)
        np.save(outname + '/post_g0', y_data_post_g0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outname', type=str, help='Name for model directory')
    parser.add_argument('--seed', type=int, default=1235,
                        help='Random seed')
    parser.add_argument('--max_sessions', type=int, default=10,
                        help='Max number of sessions to load')
    parser.add_argument('--session_type', default='recording', type=str,
                        choices=['recording', 'injection'],
                        help='Type of data session. Either neural recording ' +
                             'session or saline/muscimol injection session')
    parser.add_argument('--session_index', type=str,
                        default='~/data/penaltykick/model_data/session_index.csv',
                        help='Location of session index file')
    parser.add_argument('--data_loc', type=str,
                        default='~/data/penaltykick/model_data/compiled_penalty_kick_wspikes_wgaze_resplined.hdf5',
                        help='Location of data file')
    parser.add_argument('--rec_lag', type=int, default=10,
                        help='Number of timepoints to include as input to recognition model')
    parser.add_argument('--nlayers_rec', type=int, default=3,
                        help='Number of layers in recognition model NNs')
    parser.add_argument('--hidden_dim_rec', type=int, default=25,
                        help='Number of hidden units in recognition model NNs')
    parser.add_argument('--add_accel', action='store_true',
                        help='Add acceleration to states')
    parser.add_argument('--penalty_eps', type=float, default=1e5,
                        help='Penalty on epsilon (control signal noise)')
    parser.add_argument('--penalty_sigma', type=float, default=1e3,
                        help='Penalty on sigma (goal state noise)')
    parser.add_argument('--penalty_g', type=float, default=1e3,
                        help='Penalty for goal states escaping boundary_g.')
    parser.add_argument('--boundary_g', type=float, default=1.0,
                        help='Goal state boundary that corresponds to penalty')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for adam optimizer')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Number of iterations through the full training set')
    args = parser.parse_args()

    run_model(**vars(args))

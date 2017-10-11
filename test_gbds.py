import argparse
from tf_gbds.utils import *
# from tf_gbds.prep_GAN import prep_GAN_data, prep_injection_conditions
import sys
import os
from os.path import expanduser
from tf_gbds.PenaltyKick import SGVB_GBDS
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time
# import pickle as pkl
# export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH


def gen_data(n_trial, n_obs, sigma=np.log1p(np.exp(-10 * np.zeros((1, 2)))),
             Kp=1, Ki=0, Kd=0, vel=np.array([1e-3, 1e-2, 1e-2], np.float32)):
    data = []

    for s in range(n_trial):
        p_b = np.zeros((n_obs, 2), np.float32)
        p_g = np.zeros((n_obs, 1), np.float32)
        g = np.zeros(p_b.shape, np.float32)
        g[0] = [0.5, 0.5]
        prev_error_b = 0
        prev_error_g = 0
        int_error_b = 0
        int_error_g = 0

        for t in range(n_obs - 1):
            if p_b[t, 0] > 0.3:
                next_g_mu = np.array([0.5, -0.5], np.float32)
                next_g_lambda = np.array([36, 36], np.float32)
            else:
                next_g_mu = np.array([0.25, 0.25], np.float32)
                next_g_lambda = np.array([16, 16], np.float32)

            g[t + 1] = (g[t] + next_g_lambda * next_g_mu) / (1 + next_g_lambda)
            var = sigma ** 2 / (1 + next_g_lambda)
            g[t + 1] += (np.random.randn(1, 2) * np.sqrt(var)).reshape(2,)

            error_b = g[t + 1] - p_b[t]
            int_error_b += error_b
            der_error_b = error_b - prev_error_b
            u_b = Kp * error_b + Ki * int_error_b + Kd * der_error_b
            prev_error_b = error_b
            p_b[t + 1] = p_b[t] + vel[1:] * np.tanh(u_b)

            error_g = p_b[t + 1, 1] - p_g[t]
            int_error_g += error_g
            der_error_g = error_g - prev_error_g
            u_g = Kp * error_g + Ki * int_error_g + Kd * der_error_g
            prev_error_g = error_g
            p_g[t + 1] = p_g[t] + vel[0] * np.tanh(u_g)

        data.append(np.hstack([p_g, p_b]))

    return data


def run_model(**kwargs):
    outname = kwargs['outname']
    seed = kwargs['seed']
    # max_sessions = kwargs['max_sessions']
    # session_type = kwargs['session_type']
    # session_index = kwargs['session_index']
    # data_loc = kwargs['data_loc']
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
    # if session_type == 'recording':
    #     print 'Loading movement data from recording sessions...'
    #     groups = get_session_names(session_index,
    #                                ('type',), ('recording',))
    # elif session_type == 'injection':
    #     print 'Loading movement data from injection sessions...'
    #     groups = get_session_names(session_index,
    #                                ('type', 'type'),
    #                                ('saline', 'muscimol'),
    #                                comb=np.any)
    # sys.stdout.flush()
    # groups = groups[-max_sessions:]
    # y_data, y_data_modes, y_val_data, y_val_data_modes = load_pk_data(data_loc,
    #                                                                   groups)
    data = gen_data(n_trial=100, n_obs=200, Kp=2, Ki=0.1, Kd=0.5)

    train_data = []
    val_data = []
    for trial in data:
        if np.random.rand() <= 0.85:
            train_data.append(trial)
        else:
            val_data.append(trial)
    np.save(outname + '/train_data', train_data)
    np.save(outname + '/val_data', val_data)

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

    with tf.name_scope('model_setup'):
        with tf.name_scope('rec_params'):
            rec_params = get_rec_params_GBDS(seed, obs_dim, rec_lag, nlayers_rec,
                                             hidden_dim_rec, penalty_Q,
                                             PKLparams)

        with tf.name_scope('gen_params'):
            with tf.name_scope('gen_goalie_params'):
                gen_params_g = get_gen_params_GBDS(seed, obs_dim_g, obs_dim, add_accel,
                                                   yCols_goalie, nlayers_rec,
                                                   hidden_dim_rec, PKLparams, vel,
                                                   penalty_eps, penalty_sigma,
                                                   boundaries_g, penalty_g, 'goalie')

            with tf.name_scope('gen_ball_params'):
                gen_params_b = get_gen_params_GBDS(seed, obs_dim_b, obs_dim, add_accel,
                                                   yCols_ball, nlayers_rec, hidden_dim_rec,
                                                   PKLparams, vel, penalty_eps,
                                                   penalty_sigma, boundaries_g, penalty_g, 'ball')

        model = SGVB_GBDS(gen_params_b, gen_params_g, yCols_ball, yCols_goalie,
                          rec_params, ntrials)

        with tf.name_scope('cost'):
            cost = -model.cost()


    print('Check params:')
    if add_pklayers:
        if row_sparse:
            print('a: %f' % model.all_PKbias_layers[0].a)
            print('b: %f' % model.all_PKbias_layers[0].b)
        else:
            print('k: %f' % model.all_PKbias_layers[0].k)
    print('--------------Generative Params----------------')
    if penalty_eps is not None:
        print('Penalty on control signal noise, epsilon (Generative): %i' % penalty_eps)
    if penalty_sigma is not None:
        print('Penalty on goal state noise, sigma (Generative): %i' % penalty_sigma)
    if penalty_g[0] is not None:
        print('Penalty on goal state leaving boundary 1 (Generative): %i' % penalty_g[0])
    if penalty_g[1] is not None:
        print('Penalty on goal state leaving boundary 2 (Generative): %i' % penalty_g[1])
    print('--------------Recognition Params---------------')
    print('Num Layers (VILDS recognition): %i' % nlayers_rec)
    print('Hidden Dims (VILDS recognition): %i' % hidden_dim_rec)
    print('Input lag (VILDS recognition): %i' % rec_lag)
    # sys.stdout.flush()

    # set up an iterator over our training data
    data_iter_vb = DatasetTrialIndexIterator(train_data, randomize=True)
    # data_iter_vb = DatasetMiniBatchIterator(train_data, batch_size=50, randomize=True)

    val_costs = []
    ctrl_cost = []

    print('Setting up VB model...')
    # sys.stdout.flush()
    # ctrl_train_fn, ctrl_test_fn = (
    #     compile_functions_vb_training(model, learning_rate,
    #                                   add_pklayers=add_pklayers,
    #                                   mode=COMPILE_MODE))
    # if add_pklayers:
    #     model.mode = tf.placeholder(tf.float32, name='batch_mode')
    # if joint_spikes and model.isTrainingSpikeModel:
    #     model.spikes = tf.placeholder(tf.float32, name='batch_spikes')
    #     model.signals = tf.placeholder(tf.float32, name='batch_signals')

    # cap_noise = False

    with tf.name_scope('train'):
        model.set_training_mode('CTRL')
        opt = tf.train.AdamOptimizer(learning_rate)
        train_op = opt.minimize(cost, var_list=model.getParams())

    tf.summary.scalar('train_cost', cost)

    tf.summary.scalar('Kp_x_ball', model.mprior_ball.Kp[0, 0])
    tf.summary.scalar('Kp_y_ball', model.mprior_ball.Kp[1, 0])
    tf.summary.scalar('Ki_x_ball', model.mprior_ball.Ki[0, 0])
    tf.summary.scalar('Ki_y_ball', model.mprior_ball.Ki[1, 0])
    tf.summary.scalar('Kd_x_ball', model.mprior_ball.Kd[0, 0])
    tf.summary.scalar('Kd_y_ball', model.mprior_ball.Kd[1, 0])

    summary_op = tf.summary.merge_all()

    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    saver = tf.train.Saver()

    # Iterate over the training data for the specified number of epochs
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    # with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(outname, sess.graph)

        print('---> Training control model')
        # sys.stdout.flush()
        for ie in range(n_epochs):
            print('--> entering epoch %i' % (ie + 1))
            # sys.stdout.flush()
            # start_train = time.time()
            # step = 0
            for data in data_iter_vb:
                # step += 1
                # if step % 40 == 0:
                #     _, cost, train_summary = sess.run([train_op, model.cost(), summary_op],
                #                                       feed_dict={model.Y: data},
                #                                       options=run_options,
                #                                       run_metadata=run_metadata)
                #     ctrl_cost.append(cost)
                #     # train_writer.add_run_metadata(run_metadata, 'epoch_%s_step_%s' % (ie + 1, step))
                #     train_writer.add_summary(train_summary)
                #     # ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
                #     # opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()).with_node_names(show_name_regexes=['.*test_gbds.py.*']).build()

                #     # tf.profiler.profile(tf.get_default_graph(),
                #     #                     run_meta=run_metadata,
                #     #                     cmd='code',
                #     #                     options=opts)
                # else:
                _, train_cost, train_summary = sess.run([train_op, cost, summary_op],
                                                        feed_dict={model.Y: data})
                ctrl_cost.append(train_cost)
                train_writer.add_summary(train_summary)

                # for param in model.getParams():
                #     if param.name == 'W' or param.name == 'b' or param.name == 'G':
                #         # only on NN parameters
                #         param = tf.clip_by_norm(param, 5, axes=[0])
                #     if cap_noise and param.name == 'QinvChol' or param.name == 'Q0invChol':
                #         param = tf.clip_by_norm(param, 30, axes=[0])
            # print('Epoch %i takes %0.3f s' % ((ie + 1), (time.time() - start_train)))

            # start_val = time.time()
            curr_val_costs = []
            for i in range(len(val_data)):
                val_cost = sess.run(cost, feed_dict={model.Y: val_data[i]})
                curr_val_costs.append(val_cost)
            val_costs.append(np.array(curr_val_costs).mean())
            # print('Validation step takes %0.3f s' % (time.time() - start_val))
            print('Validation set cost: %f' % val_costs[-1])
            # # sys.stdout.flush()
            # with open(outname + '/model.pkl', 'wb') as f:
            #     pkl.dump(model, f)
            # if ie == 0:
            #     saver.save(sess, outname + '/gbds_test', write_meta_graph=True)
            if ie == (n_epochs - 1):
                saver.save(sess, outname + '/gbds_test', write_meta_graph=True)
            np.save(outname + '/train_costs', ctrl_cost)
            np.save(outname + '/val_costs', val_costs)

        train_writer.close()

        # save outputs to use during CGAN and GAN training
        # print('----> Preparing GAN data')
        # sys.stdout.flush()
        # # add injection location and type as conditions to cGAN
        # inject_conds = None
        # # y_data_postJ, y_data_post_g0, y_data_states = prep_GAN_data(
        # #     model, train_data, extra_cond=inject_conds)
        # all_postJ = []
        # all_post_g0 = []
        # all_states = []
        # for i in range(len(train_data)):
        #     trial = train_data[i]
        #     g_samp = tf.squeeze(model.mrec.getSample(), 2).eval({model.Y: trial})
        #     trial_postJ = np.hstack(
        #         [model.mprior_goalie.draw_postJ(
        #              tf.gather(g_samp, model.mprior_goalie.yCols, axis=1)).eval(),
        #          model.mprior_ball.draw_postJ(
        #              tf.gather(g_samp, model.mprior_ball.yCols, axis=1)).eval()])
        #     trial_post_g0 = g_samp[0]
        #     trial_states = model.mprior_ball.get_states(trial).eval()
        #     all_postJ.append(trial_postJ)
        #     all_post_g0.append(trial_post_g0)
        #     # don't have postJ for last index since it needs g at t+1
        #     # so ignore state at last index
        #     curr_states = trial_states[:-1]
        #     if inject_conds is not None:  # if additional conditions are provided
        #         curr_states = np.hstack((curr_states, inject_conds[i][:-1]))
        #     all_states.append(curr_states)

        # all_postJ = np.vstack(all_postJ)
        # all_post_g0 = np.vstack(all_post_g0)
        # all_states = np.vstack(all_states)
        # np.save(outname + '/postJ', all_postJ)
        # np.save(outname + '/states', all_states)
        # np.save(outname + '/post_g0', all_post_g0)

        Kp_b = model.mprior_ball.Kp.eval()
        Ki_b = model.mprior_ball.Ki.eval()
        Kd_b = model.mprior_ball.Kd.eval()
        # pid_b = [Kp_b, Ki_b, Kd_b]
        # np.save(outname + '/pid_ball', pid_b)
        # eps_b = model.mprior_ball.eps.eval()

    print('----> Predicting trajectory')
    # sys.stdout.flush()
    n_obs = train_data[0].shape[0]
    pred = np.zeros((n_obs, 2), np.float32)
    g_ball = np.zeros((n_obs, 2), np.float32)
    g_ball[0] = [0.5, 0.5]
    sigma = np.log1p(np.exp(-10 * np.zeros((1, 2))))

    prev_error_b = 0
    int_error_b = 0

    for t in range(n_obs - 1):
        if pred[t, 0] > 0.3:
            next_g_mu = np.array([0.5, -0.5], np.float32)
            next_g_lambda = np.array([36, 36], np.float32)
        else:
            next_g_mu = np.array([0.25, 0.25], np.float32)
            next_g_lambda = np.array([16, 16], np.float32)

        g_ball[t + 1] = (g_ball[t] + next_g_lambda * next_g_mu) / (1 + next_g_lambda)
        var = sigma ** 2 / (1 + next_g_lambda)
        g_ball[t + 1] += (np.random.randn(1, 2) * np.sqrt(var)).reshape(2,)

        error_b = g_ball[t + 1] - pred[t]
        int_error_b += error_b
        der_error_b = error_b - prev_error_b
        u_b = Kp_b.reshape(2,) * error_b + Ki_b.reshape(2,) * int_error_b + Kd_b.reshape(2,) * der_error_b
        prev_error_b = error_b
        # pred[t + 1] = pred[t] + vel[1:] * np.tanh(u_b + eps_b.reshape(2,) * np.random.randn(2,))
        pred[t + 1] = pred[t] + vel[1:] * np.tanh(u_b)

    np.save(outname + '/pred_trajectory', pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outname', type=str, default='model_saved',
                        help='Name for model directory')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    # parser.add_argument('--max_sessions', type=int, default=10,
    #                     help='Max number of sessions to load')
    # parser.add_argument('--session_type', default='recording', type=str,
    #                     choices=['recording', 'injection'],
    #                     help='Type of data session. Either neural recording ' +
    #                          'session or saline/muscimol injection session')
    # parser.add_argument('--session_index', type=str,
    #                     default='~/data/penaltykick/model_data/session_index.csv',
    #                     help='Location of session index file')
    # parser.add_argument('--data_loc', type=str,
    #                     default='~/data/penaltykick/model_data/compiled_penalty_kick_wspikes_wgaze_resplined.hdf5',
    #                     help='Location of data file')
    parser.add_argument('--rec_lag', type=int, default=10,
                        help='Number of timepoints to include as input to recognition model')
    parser.add_argument('--nlayers_rec', type=int, default=3,
                        help='Number of layers in recognition model NNs')
    parser.add_argument('--hidden_dim_rec', type=int, default=25,
                        help='Number of hidden units in recognition model NNs')
    parser.add_argument('--add_accel', action='store_true',
                        help='Add acceleration to states')
    parser.add_argument('--penalty_eps', type=float, default=None,
                        help='Penalty on epsilon (control signal noise)')
    parser.add_argument('--penalty_sigma', type=float, default=None,
                        help='Penalty on sigma (goal state noise)')
    parser.add_argument('--penalty_g', type=float, default=None,
                        help='Penalty for goal states escaping boundary_g.')
    parser.add_argument('--boundary_g', type=float, default=1.0,
                        help='Goal state boundary that corresponds to penalty')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for adam optimizer')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of iterations through the full training set')
    args = parser.parse_args()

    run_model(**vars(args))

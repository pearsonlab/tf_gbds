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
from tensorflow.python.client import timeline
# import pickle as pkl
# export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH


def gen_data(n_trial, n_obs, sigma=np.log1p(np.exp(-5 * np.ones((1, 2)))),
             eps=1e-5, Kp=1, Ki=0, Kd=0,
             vel=1e-2 * np.ones((3))):
    p = []
    g_b = []

    for s in range(n_trial):
        p_b = np.zeros((n_obs, 2), np.float32)
        p_g = np.zeros((n_obs, 1), np.float32)
        g = np.zeros(p_b.shape, np.float32)
        prev_error_b = 0
        prev_error_g = 0
        int_error_b = 0
        int_error_g = 0

        init = np.pi * (np.random.rand() * 2 - 1)
        g_mu_y = 0.75 * np.sin(1. * (np.linspace(0, 2 * np.pi, n_obs) - init))
        g_mu = np.hstack([np.ones((n_obs, 1)), g_mu_y.reshape(n_obs, 1)])
        g_lambda = np.array([16, 16], np.float32)
        g[0] = [1, 0.75 * (np.random.rand() * 2 - 1)]

        for t in range(n_obs - 1):
            g[t + 1] = (g[t] + g_lambda * g_mu[t + 1]) / (1 + g_lambda)
            var = sigma ** 2 / (1 + g_lambda)
            g[t + 1] += (np.random.randn(1, 2) * np.sqrt(var)).reshape(2,)

            error_b = g[t + 1] - p_b[t]
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

        p.append(np.hstack([p_g, p_b]))
        g_b.append(g)

    return p, g_b

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
    tol = 1e-6
    eta = tf.Variable(initial_value=5e-2, trainable=False, name='eta',
                      dtype=tf.float32)
    eta_val = np.exp(np.log(5e-2) - np.arange(n_epochs) / 20)
    eta_val[eta_val < 1e-7] = 1e-7

    data, goals = gen_data(n_trial=2000, n_obs=100, Kp=0.8, Ki=0.4, Kd=0.2)
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

    with tf.name_scope('model_setup'):
        with tf.name_scope('rec_goal_params'):
            rec_params_goal = get_rec_params_GBDS('goal', obs_dim, rec_lag,
                                                  nlayers_rec, hidden_dim_rec,
                                                  penalty_Q, PKLparams)
        with tf.name_scope('rec_control_signal_params'):
            rec_params_ctrl = get_rec_params_GBDS('ctrl', obs_dim, rec_lag,
                                                  nlayers_rec, hidden_dim_rec,
                                                  penalty_Q, PKLparams)

        with tf.name_scope('gen_goalie_params'):
            gen_params_g = get_gen_params_GBDS('goalie', obs_dim_g, obs_dim,
                                               add_accel, yCols_goalie,
                                               nlayers_rec, hidden_dim_rec,
                                               PKLparams, vel, penalty_eps,
                                               penalty_sigma, boundaries_g,
                                               penalty_g)
        with tf.name_scope('gen_ball_params'):
            gen_params_b = get_gen_params_GBDS('ball', obs_dim_b, obs_dim,
                                               add_accel, yCols_ball,
                                               nlayers_rec, hidden_dim_rec,
                                               PKLparams, vel, penalty_eps,
                                               penalty_sigma, boundaries_g,
                                               penalty_g)

        model = SGVB_GBDS(gen_params_b, gen_params_g, yCols_ball, yCols_goalie,
                          rec_params_goal, rec_params_ctrl, ntrials)

        with tf.name_scope('cost'):
            with tf.name_scope('goal_samples'):
                q_g_mean = tf.squeeze(model.mrec_goal.postX, 2,
                                      name='q_g_mean')
                q_g = tf.squeeze(model.mrec_goal.samples, 2, name='q_g')
            with tf.name_scope('control_signal_samples'):
                q_U_mean = tf.squeeze(model.mrec_ctrl.postX, 2,
                                      name='q_U_mean')
                q_U = tf.squeeze(model.mrec_ctrl.samples, 2, name='q_U')

            with tf.name_scope('gen_goalie_logdensity'):
                gen_goalie_logdensity = model.mprior_goalie.evaluateLogDensity(
                    tf.gather(q_g, model.yCols_goalie, axis=1),
                    tf.gather(q_U, model.yCols_goalie, axis=1),
                    model.Y, tol, eta)
            with tf.name_scope('gen_ball_logdensity'):
                gen_ball_logdensity = model.mprior_ball.evaluateLogDensity(
                    tf.gather(q_g, model.yCols_ball, axis=1),
                    tf.gather(q_U, model.yCols_ball, axis=1),
                    model.Y, tol, eta)
            with tf.name_scope('rec_goal_entropy'):
                rec_goal_entropy = model.mrec_goal.evalEntropy()
            with tf.name_scope('rec_ctrl_entropy'):
                rec_ctrl_entropy = model.mrec_ctrl.evalEntropy()

            cost = -((gen_goalie_logdensity + gen_ball_logdensity +
                      rec_goal_entropy + rec_ctrl_entropy) /
                     tf.cast(tf.shape(model.Y)[0], tf.float32))

        # _, _, U_pred_g, _ = model.mprior_goalie.get_preds(
        #     model.Y[:-1], training=True,
        #     post_g=tf.gather(q_g, yCols_goalie, axis=1),
        #     post_U=tf.gather(q_U, yCols_goalie, axis=1))
        # _, _, U_pred_b, _ = model.mprior_ball.get_preds(
        #     model.Y[:-1], training=True,
        #     post_g=tf.gather(q_g, yCols_ball, axis=1),
        #     post_U=tf.gather(q_U, yCols_ball, axis=1))
        # U_pred = tf.concat([U_pred_g, U_pred_b], axis=1)

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

    val_costs = []
    ctrl_cost = []

    print('Setting up VB model...')
    # sys.stdout.flush()
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
    tf.summary.scalar('rec_goal_entropy', rec_goal_entropy)
    tf.summary.scalar('rec_ctrl_entropy', rec_ctrl_entropy)
    tf.summary.scalar('gen_goalie_logdensity', gen_goalie_logdensity)
    tf.summary.scalar('gen_ball_logdensity', gen_ball_logdensity)
    tf.summary.scalar('Kp_x_ball', model.mprior_ball.Kp[0, 0])
    tf.summary.scalar('Kp_y_ball', model.mprior_ball.Kp[1, 0])
    tf.summary.scalar('Ki_x_ball', model.mprior_ball.Ki[0, 0])
    tf.summary.scalar('Ki_y_ball', model.mprior_ball.Ki[1, 0])
    tf.summary.scalar('Kd_x_ball', model.mprior_ball.Kd[0, 0])
    tf.summary.scalar('Kd_y_ball', model.mprior_ball.Kd[1, 0])
    tf.summary.scalar('eta', eta)

    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    # Iterate over the training data for the specified number of epochs
    with tf.Session() as sess:
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

                _, train_cost, train_summary = sess.run(
                    [train_op, cost, summary_op],
                    feed_dict={model.Y: data, eta: eta_val[ie]})
                ctrl_cost.append(train_cost)
                train_writer.add_summary(train_summary)

                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                # sess.run(train_op, feed_dict={model.Y: data},
                #          options=run_options,
                #          run_metadata=run_metadata)

                # tf.profiler.profile(tf.get_default_graph(),
                #                     run_meta=run_metadata,
                #                     cmd='scope',
                #                     options=tf.profiler.ProfileOptionBuilder.time_and_memory())

                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # with open(outname + '/timeline_%s.json' % step, 'w') as f:
                #     f.write(chrome_trace)

            # print('Epoch %i takes %0.3f s' % ((ie + 1), (time.time() - start_train)))

            # start_val = time.time()
            curr_val_costs = []
            for i in range(len(val_data)):
                val_cost = sess.run(cost, feed_dict={model.Y: val_data[i],
                                                     eta: eta_val[ie]})
                curr_val_costs.append(val_cost)
            val_costs.append(np.array(curr_val_costs).mean())
            # # print('Validation step takes %0.3f s' % (time.time() - start_val))
            print('Validation set cost: %f' % val_costs[-1])
            # # sys.stdout.flush()
            # with open(outname + '/model.pkl', 'wb') as f:
            #     pkl.dump(model, f)
            # if ie == 0:
            #     saver.save(sess, outname + '/gbds_test', write_meta_graph=True)
            np.save(outname + '/train_costs', ctrl_cost)
            np.save(outname + '/val_costs', val_costs)

            if (ie + 1) % 10 == 0:
                print('----> Predicting control signals and goals')
                ctrl_post_mean = []
                ctrl_post_samp = []
                goal_post_mean = []

                for i in range(len(val_data)):
                    ctrl_post_mean.append(
                        q_U_mean.eval(feed_dict={model.Y: val_data[i]}))

                    u_post_samp = []
                    for _ in range(30):
                        u_post_samp.append(
                            q_U.eval(feed_dict={model.Y: val_data[i]}))
                    ctrl_post_samp.append(u_post_samp)

                    goal_post_mean.append(
                        q_g_mean.eval(feed_dict={model.Y: val_data[i]}))

                np.save(outname + '/ctrl_post_mean_step_%s' % (ie + 1),
                        ctrl_post_mean)
                np.save(outname + '/ctrl_post_samp_step_%s' % (ie + 1),
                        ctrl_post_samp)
                np.save(outname + '/goal_post_mean_step_%s' % (ie + 1),
                        goal_post_mean)

                # print('----> Predicting trajectory')
                # Kp_b = model.mprior_ball.Kp.eval()
                # Ki_b = model.mprior_ball.Ki.eval()
                # Kd_b = model.mprior_ball.Kd.eval()
                # eps_b = model.mprior_ball.eps.eval()
                # sigma_b = model.mprior_ball.sigma.eval()
                # n_obs = train_data[0].shape[0]
                # pred_summary = []

                # for i in range(100):
                #     pred = np.zeros((n_obs, 2), np.float32)
                #     g_ball = np.zeros((n_obs, 2), np.float32)
                #     g_ball[0] = [0.75, 0.75]

                #     prev_error_b = 0
                #     int_error_b = 0

                #     for t in range(n_obs - 1):
                #         if np.random.uniform() >= 0.5:
                #             next_g_mu = np.array([0.75, -0.75], np.float32)
                #             next_g_lambda = np.array([1, 1], np.float32)
                #         else:
                #             next_g_mu = np.array([0.75, 0.75], np.float32)
                #             next_g_lambda = np.array([1, 1], np.float32)

                #         g_ball[t + 1] = (g_ball[t] + next_g_lambda * next_g_mu) / (1 + next_g_lambda)
                #         var = sigma_b ** 2 / (1 + next_g_lambda)
                #         g_ball[t + 1] += (np.random.randn(1, 2) * np.sqrt(var)).reshape(2,)

                #         error_b = g_ball[t + 1] - pred[t]
                #         int_error_b += error_b
                #         der_error_b = error_b - prev_error_b
                #         u_b = Kp_b.reshape(2,) * error_b + Ki_b.reshape(2,) * int_error_b + Kd_b.reshape(2,) * der_error_b
                #         prev_error_b = error_b
                #         pred[t + 1] = pred[t] + vel[1:] * np.tanh(u_b + (eps_b * np.random.randn(2)).reshape(2,))
                #     pred_summary.append(pred)

                # np.save(outname + '/pred_trajectory_step_%s' % (ie + 1), pred_summary)

            if ie == (n_epochs - 1):
                saver.save(sess, outname + '/gbds_test', write_meta_graph=True)

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
    parser.add_argument('--n_epochs', type=int, default=500,
                        help='Number of iterations through the full training set')
    args = parser.parse_args()

    os.environ['CUDA_​VISIBLE_​DEVICES'] = ''
    run_model(**vars(args))

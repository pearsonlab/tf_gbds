"""
Assemble data for large scale GAN training

Input:
Model with VB training complete (untrained GAN)

Output:
* posterior on J (mu and sigma), all trials stacked into one matrix
* corresponding states, s, all trials stacked into one matrix
* posterior on initial goal state for each trial, all stacked into one matrix
"""
import argparse
import tensorflow as tf
import sys
import pickle as pkl
import numpy as np
from tf_gbds.utils import get_session_names, load_pk_data
from os.path import join, expanduser

# sys.path.append(expanduser('~/code/gbds/code/'))
# sys.path.append(expanduser('~/code/gbds/code/lib/'))


def prep_injection_conditions(data, modes):
    """
    Given trajectory data (used for shape information only) and modes (look in
    utils.load_pk_data for more info), prepare conditions for GAN:

    [0, 0, 0, 0]: normal
    [1, 0, 0, 0]: saline and DLPFC
    [0, 1, 0, 0]: saline and DMPFC
    [1, 0, 1, 0]: muscimol and DLPFC
    [0, 1, 0, 1]: muscimol and DMPFC
    """
    inject_conds = []
    for i in range(len(data)):
        trial_len = len(data[i])
        curr_mode = modes[i].reshape((1, -1)).astype(np.float32)
        inject_conds.append(np.repeat(curr_mode, trial_len, axis=0))
    return inject_conds


def prep_GAN_data(model_in, data, extra_cond=None):
    """
    model_in: directory filepath or model object
    data: positional data to be transformed into GAN inputs. Shape:
          (n_trials) with each trial having shape: (n_timepoints, n_dimensions)
    extra_cond: Additional conditions (other than states) to include as input
                to cGAN. Shape: (n_trials,) with each trial having shape:
                (n_timepoints, n_conditions)
    """
    # if type(model_in) is str:  # directory filepath
    #     with open(join(model_in, 'model.pkl'), 'rb') as f:
    #         model = pkl.load(f)
    # else:
    #     model = model_in
    model = model_in

    all_postJ = []
    all_post_g0 = []
    all_states = []
    for i in range(len(data)):
        trial = data[i]
        g_samp = tf.squeeze(model.mrec.getSample(), 2)
        trial_postJ = tf.concat(
            [model.mprior_goalie.draw_postJ(
                 tf.gather(g_samp, model.mprior_goalie.yCols, axis=1)),
             model.mprior_ball.draw_postJ(
                 tf.gather(g_samp, model.mprior_ball.yCols, axis=1))],
            axis=1).eval()
        trial_post_g0 = g_samp[0].eval()
        trial_states = model.mprior_ball.get_states([trial]).eval()
        all_postJ.append(trial_postJ)
        all_post_g0.append(trial_post_g0)
        # don't have postJ for last index since it needs g at t+1
        # so ignore state at last index
        curr_states = trial_states[:-1]
        if extra_cond is not None:  # if additional conditions are provided
            curr_states = np.hstack((curr_states, extra_cond[i][:-1]))
        all_states.append(curr_states)

    all_postJ = np.vstack(all_postJ)
    all_post_g0 = np.vstack(all_post_g0)
    all_states = np.vstack(all_states)

    return all_postJ, all_post_g0, all_states


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('directory', help='Name of directory for model')
#     parser.add_argument('--seed', type=int, default=1235,
#                         help='Random seed')
#     parser.add_argument('--max_sessions', type=int, default=10,
#                         help='Max number of sessions to load')
#     parser.add_argument('--session_type', default='recording', type=str,
#                         choices=['recording', 'injection'],
#                         help='Type of data session. Either neural recording ' +
#                              'session or saline/muscimol injection session')
#     parser.add_argument('--session_index', type=str,
#                         default='~/data/penaltykick/model_data/session_index.csv',
#                         help='Location of session index file')
#     parser.add_argument('--data_loc', type=str,
#                         default='~/data/penaltykick/model_data/compiled_penalty_kick_wspikes_wgaze_resplined.hdf5',
#                         help='Location of data file')
#     args = parser.parse_args()

#     np.random.seed(args.seed)  # set seed for consistent train/val split

#     if args.session_type == 'recording':
#         print("Loading movement data from recording sessions...")
#         groups = get_session_names(args.session_index,
#                                    ('type',), ('recording',))
#     elif args.session_type == 'injection':
#         print("Loading movement data from injection sessions...")
#         groups = get_session_names(args.session_index,
#                                    ('type', 'type'),
#                                    ('saline', 'muscimol'),
#                                    comb=np.any)
#     sys.stdout.flush()
#     groups = groups[-args.max_sessions:]
#     y_data, y_data_modes, y_val_data, y_val_data_modes = load_pk_data(
#         args.data_loc, groups)

#     print("Preparing GAN data...")
#     sys.stdout.flush()
#     # add injection location and type as conditions to cGAN
#     if args.session_type == 'injection':
#         inject_conds = prep_injection_conditions(y_data, y_data_modes)
#     else:
#         inject_conds = None
#     y_data_postJ, y_data_post_g0, y_data_states = prep_GAN_data(
#         args.directory, y_data, extra_cond=inject_conds)
#     np.save(join(args.directory, 'postJ'), y_data_postJ)
#     np.save(join(args.directory, 'states'), y_data_states)
#     np.save(join(args.directory, 'post_g0'), y_data_post_g0)

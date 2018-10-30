"""
Utility functions for generating prey trajectory.
Pre-requisite: the cost grid (tf.Tensor) defining the pacman playground.
"""

import numpy as np
import tensorflow as tf

# constants
npc_size = 30
lim_xscreen = 1920
lim_yscreen = 1080
min_screen = 0
max_screen = np.array([lim_xscreen, lim_yscreen])
num_angles = 15
min_dist = 400  # Minimal initial distance
max_dist = np.sqrt((lim_xscreen - 2 * npc_size) ** 2 +
                   (lim_yscreen - 2 * npc_size) ** 2)
dist_factor = 50
npc_factor = 50


def generate_weight(axis_start, axis_end, axis_res, x_shift, slope):
    axis_val = np.linspace(axis_start, axis_end, axis_res)

    return 1. / (1. + np.exp(-slope * (axis_val - x_shift)))


def generate_rotation_mat():
    """Generates rotation vectors as a matrix (called once).
    """
    rot_mat = []
    for iRot in range(0, num_angles):
        theta = iRot * (2 * np.pi / num_angles)
        rot_vect = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        rot_mat.append(rot_vect.astype(np.float32))

    return rot_mat


def recover_orig_val(data):
    """Converts normalized coordinates to original scale (pixels on screen).
    Input data is normalized and smoothed.
    """
    return max_screen * (data + 1.) / 2


def recover_normalized(data):
    """Normalizes original coordinates.
    """
    return 2 * data / max_screen - 1.


def generate_ps_vect(prey_pos, subj_pos, npc_spd, weight_x, weight_y):
    """Generates the vector between prey and subject.
    """
    orig_vect = tf.subtract(prey_pos, subj_pos, "original")
    norm_vect = tf.divide(orig_vect, tf.norm(orig_vect), "normalized")

    # Weights are unequally distributed along y and x axes (screen is 4:3).
    with tf.name_scope("reweight"):
        def f1_x(): return weight_x[-1]
        def f2_x(): return weight_x[0]
        def f3_x(): return tf.gather(
            weight_x, tf.to_int64(tf.floor(tf.abs(orig_vect[0]))),
            name="x_index")

        w_x = tf.case({tf.greater_equal(orig_vect[0], lim_xscreen): f1_x,
                       tf.less_equal(orig_vect[0], 0): f2_x},
                      default=f3_x, exclusive=False, name="x_weight")

        def f1_y(): return weight_y[-1]
        def f2_y(): return weight_y[0]
        def f3_y(): return tf.gather(
            weight_y, tf.to_int64(tf.floor(tf.abs(orig_vect[1]))),
            name="y_index")

        w_y = tf.case({tf.greater_equal(orig_vect[1], lim_yscreen): f1_y,
                       tf.less_equal(orig_vect[1], 0): f2_y},
                      default=f3_y, exclusive=False, name="y_weight")

        real_vect = tf.multiply(norm_vect, tf.to_float([w_x, w_y]) * npc_spd,
                                "reweighted")

    return real_vect


def compute_movement_angle(prey_pos, subj_pos, real_vect, cost_grid, rot_mat):
    """Scans through angles and finds next movement of prey.
    """
    tmp_vect = []
    ply_cost = []
    ls_cost = []

    for iAng in range(0, num_angles):
        # Originally away from player (vector 180 degree away from player)
        #   Then apply rotation vector for generating rotated place.
        tmp_vect_i = tf.squeeze(
            tf.matmul(rot_mat[iAng], tf.reshape(real_vect, [2, 1])))
        tmp_vect.append(tmp_vect_i)

        # Following two conditions some kind of clipping the range where npc might go
        #   (x: not below npc_size and beyond 1920-npc_size
        #   || y: not below npc_size and beyond 1080-npc_size).
        ang_pos_i = tf.clip_by_value(
            tf.floor(prey_pos + tmp_vect_i), [npc_size, npc_size],
            [lim_xscreen - npc_size, lim_yscreen - npc_size])

        # Landscape cost computation
        ls_cost.append(
            tf.gather_nd(cost_grid, tf.to_int64([ang_pos_i[1], ang_pos_i[0]])))

        # Some more cost devices were implemented in original task.
        np_magn = tf.sqrt(tf.reduce_sum((ang_pos_i - subj_pos) ** 2))
        ply_cost.append((-np_magn + 1))

    return ls_cost, ply_cost, tmp_vect


def check_outbound(curr_pos):
    """Make sure that npc does not go out of arena.
    """
    return tf.clip_by_value(curr_pos, [npc_size, npc_size],
                            [lim_xscreen - npc_size, lim_yscreen - npc_size])


def generate_prey_trajectory(subj_pos, prey_pos, npc_spd, cost_grid,
                             weight_x, weight_y, rot_mat):
    """Generates prey position at next time step (in original coordinates).
    """
    # normalize vector between prey and subject position by prey velocity
    real_vect = generate_ps_vect(prey_pos, subj_pos, npc_spd,
                                 weight_x, weight_y)

    # define prey movement
    ls_cost, ply_cost, tmp_vect = compute_movement_angle(
        prey_pos, subj_pos, real_vect, cost_grid, rot_mat)

    # consider subject dynamic
    def f1(): return (dist_factor * (ply_cost - tf.reduce_min(ply_cost)) /
                      (tf.reduce_max(ply_cost) - tf.reduce_min(ply_cost)))
    def f2(): return (dist_factor * (ply_cost - tf.reduce_min(ply_cost)) /
                      0.00001)

    subj_cost = tf.cond(
        tf.equal(tf.reduce_max(ply_cost), tf.reduce_min(ply_cost)), f2, f1)
    min_cost_ind = tf.argmin(ls_cost + subj_cost)

    # generate next position
    prey_pos = prey_pos + tf.gather(tmp_vect, min_cost_ind)
    prey_pos = check_outbound(prey_pos)

    return prey_pos


def compute_npc_cost(first_prey_pos, second_prey_pos, vect_subj, rot_mat):
    npc_cost = []

    for iAng in range(0, num_angles):
        ang_pos_i = tf.clip_by_value(
            tf.floor(second_prey_pos + tf.squeeze(tf.matmul(
                rot_mat[iAng], tf.reshape(vect_subj[iAng], [2, 1])))),
            [npc_size, npc_size],
            [lim_xscreen - npc_size, lim_yscreen - npc_size])

        npc_magn = tf.sqrt(
            tf.reduce_sum((ang_pos_i - first_prey_pos) ** 2))
        npc_cost.append((-npc_magn + 1))

    return npc_cost


def generate_second_prey_trajectory(subj_pos, first_prey_pos, second_prey_pos,
                                    npc_spd, cost_grid, weight_x, weight_y,
                                    rot_mat):
    """Generate (second) prey position at next time step.
    The second prey has natural repulsion to first prey (not vice versa).
    """
    # generate prey-subject vector
    second_prey_vect = generate_ps_vect(second_prey_pos, subj_pos, npc_spd,
                                        weight_x, weight_y)

    # calculate costs for next movement (15 positions considered)
    ls_cost, ply_cost, tmp_vect = compute_movement_angle(
        second_prey_pos, subj_pos, second_prey_vect, cost_grid, rot_mat)

    # consider cost related to subject position
    def f1(): return (dist_factor * (ply_cost - tf.reduce_min(ply_cost)) /
                      (tf.reduce_max(ply_cost) - tf.reduce_min(ply_cost)))
    def f2(): return (dist_factor * (ply_cost - tf.reduce_min(ply_cost)) /
                      0.00001)
    subj_cost = tf.cond(
        tf.equal(tf.reduce_max(ply_cost), tf.reduce_min(ply_cost)), f2, f1)

    # consider cost based on the distance to other prey
    npc_cost = compute_npc_cost(
        first_prey_pos, second_prey_pos, tmp_vect, rot_mat)

    def n_f1(): return (npc_factor * (npc_cost - tf.reduce_min(npc_cost)) /
                        (tf.reduce_max(npc_cost) - tf.reduce_min(npc_cost)))
    def n_f2(): return (npc_factor * (npc_cost - tf.reduce_min(npc_cost)) /
                        0.00001)
    npc_cost = tf.cond(
        tf.equal(tf.reduce_max(npc_cost), tf.reduce_min(npc_cost)), n_f2, n_f1)

    final_cost = ls_cost + subj_cost + npc_cost
    min_cost_ind = tf.argmin(final_cost)

    # generate next position
    prey_pos = second_prey_pos + tf.gather(tmp_vect, min_cost_ind)
    prey_pos = check_outbound(prey_pos)

    return prey_pos


def generate_predator_trajectory(subj_pos, pred_pos, pred_vel,
                                 weight_x, weight_y):
    """Generates predator position at next time step (in original coordinates).
    """
    # normalize vector between pred and subject position by prey velocity
    real_vect = generate_ps_vect(pred_pos, subj_pos, pred_vel,
                                 weight_x, weight_y)

    # generate next position
    pred_pos = pred_pos + real_vect
    pred_pos = check_outbound(pred_pos)

    return pred_pos

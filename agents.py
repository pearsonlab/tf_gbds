import tensorflow as tf
from GenerativeModel import joint_GBDS
from RecognitionModel import SmoothingPastLDSTimeSeries
# from utils import get_vel
from tf_generate_trial import (recover_orig_val, recover_normalized,
                               generate_weight, generate_rotation_mat,
                               generate_prey_trajectory,
                               generate_second_prey_trajectory)


class game_model(object):
    """Auxiliary class to construct the computational graph
    (define generative and recognition models, draw samples, trial completion)
    """
    def __init__(self, params, inputs, max_vel, cost_grid,
                 extra_dim=0, n_samples=50):
        with tf.name_scope(params["name"]):
            model_dim = params["model_dim"]
            obs_dim = params["obs_dim"]
            latent_u = params["latent_u"]
            clip_range = params["clip_range"]

            traj = inputs["trajectory"]
            states = inputs["states"]
            ctrl_obs = inputs["ctrl_obs"]
            extra_conds = inputs["extra_conds"]

            self.latent_vars = {}
            self.var_list = []
            self.log_vars = []

            with tf.name_scope("variable_value_shape"):
                traj_shape = traj.shape.as_list()
                if traj_shape[0] is None:
                    B = 1
                else:
                    B = traj_shape[0]
                if traj_shape[1] is None:
                    Tt = 2
                else:
                    Tt = traj_shape[1]
                if latent_u:
                    value_shape = [B, Tt, model_dim * 2]
                else:
                    value_shape = [B, Tt, model_dim]

            self.p = joint_GBDS(
                params["p_params"], model_dim, states, ctrl_obs,
                extra_conds, latent_u, name="prior",
                value=tf.zeros(value_shape))
            self.var_list += self.p.params
            self.log_vars += self.p.log_vars

            if latent_u:
                self.q = SmoothingPastLDSTimeSeries(
                    params["q_params"], traj, model_dim * 2, obs_dim,
                    extra_conds, name="recognition")
            else:
                self.q = SmoothingPastLDSTimeSeries(
                    params["q_params"], traj, model_dim, obs_dim,
                    extra_conds, name="recognition")
            self.var_list += self.q.params
            self.log_vars += self.q.log_vars

            self.latent_vars.update({self.p: self.q})

            with tf.name_scope("GMM"):
                # game state includes both position and velocity
                state = tf.placeholder(tf.float32, [None, None, obs_dim * 2],
                                       "state")
                # pad prey information
                if extra_dim != 0:
                    extra_cond = tf.placeholder(
                        tf.float32, [None, None, extra_dim],
                        "extra_condition")
                    NN_input = tf.concat([state, extra_cond], -1, "NN_input")
                with tf.name_scope("subject"):
                    a = self.p.agents[0]
                    NN_out = a.GMM_NN(NN_input)
                    a_mu = tf.reshape(
                        NN_out[:, :, :(a.K * a.dim)],
                        [tf.shape(state)[0], -1, a.K, a.dim], "mu")
                    a_lambda = tf.reshape(
                        tf.nn.softplus(NN_out[:, :, (
                            a.K * a.dim):(2 * a.K * a.dim)]),
                        [tf.shape(state)[0], -1, a.K, a.dim], "lambda")
                    a_w = tf.nn.softmax(tf.reshape(
                        NN_out[:, :, (2 * a.K * a.dim):],
                        [tf.shape(state)[0], -1, a.K]), -1, "w")

            with tf.name_scope("posterior"):
                g_q_mu = tf.identity(
                    tf.squeeze(self.q.postX, -1)[:, :, :model_dim],
                    "goal_mean")
                q_samp = self.q.sample(n_samples)
                g_q_samp = tf.identity(
                    q_samp[:, :, :, :model_dim], "goal_samples")
                if latent_u:
                    u_q_mu = tf.identity(
                        tf.squeeze(self.q.postX, -1)[:, :, model_dim:],
                        "control_mean")
                    u_q_samp = tf.identity(
                        q_samp[:, :, :, model_dim:], "control_samples")

            with tf.name_scope("update_one_step"):
                weight_x = generate_weight(0, 10, 1920, 6, 2)
                weight_y = generate_weight(0, 10, 1080, 6, 2)
                rot_mat = generate_rotation_mat()

                with tf.name_scope("subject"):
                    prev_y = tf.placeholder(
                        tf.float32, obs_dim, "previous_position")
                    curr_y = tf.placeholder(
                        tf.float32, obs_dim, "current_position")
                    v = tf.divide(curr_y - prev_y, max_vel,
                                  "current_velocity")
                    curr_s = tf.concat([curr_y, v], 0, "current_state")

                    if inputs["extra_conds"] is not None:
                        gen_extra_cond = tf.placeholder(
                            tf.float32, extra_dim, "extra_conditions")
                    else:
                        gen_extra_cond = None

                    with tf.name_scope("goal"):
                        prev_g = tf.placeholder(
                            tf.float32, model_dim, "previous")
                        curr_g = tf.identity(self.p.update_goal(
                            curr_s, prev_g, gen_extra_cond), "current")

                    with tf.name_scope("control"):
                        with tf.name_scope("error"):
                            curr_error = tf.subtract(
                                curr_g, curr_y, "current")
                            prev_error = tf.placeholder(
                                tf.float32, model_dim, "previous")
                            prev2_error = tf.placeholder(
                                tf.float32, model_dim, "previous2")
                            errors = tf.stack(
                                [prev2_error, prev_error, curr_error], 0,
                                "all")
                        prev_u = tf.placeholder(
                            tf.float32, model_dim, "previous")
                        curr_u = tf.identity(self.p.update_ctrl(
                            errors, prev_u), "current")

                    if latent_u:
                        next_y = tf.clip_by_value(
                            curr_y + max_vel * tf.clip_by_value(
                                curr_u, clip_range[:, 0], clip_range[:, 1],
                                "clip_control"), -1., 1., "next_position")
                    else:
                        next_y = tf.clip_by_value(
                            curr_y + max_vel * tf.tanh(curr_u), -1., 1.,
                            "next_position")

                with tf.name_scope("prey"):
                    # requires both cost map and current subject location
                    orig_curr_y = recover_orig_val(curr_y)

                    with tf.name_scope("first"):
                        npc_spd_1 = 10 + gen_extra_cond[4] * 4
                        curr_prey_1 = gen_extra_cond[:2]
                        next_prey_1 = tf.identity(
                            recover_normalized(generate_prey_trajectory(
                                orig_curr_y, recover_orig_val(curr_prey_1),
                                npc_spd_1, cost_grid, weight_x, weight_y,
                                rot_mat)), "next")

                    with tf.name_scope("second"):
                        def f1():
                            npc_spd_2 = 10 + gen_extra_cond[9] * 4
                            curr_prey_2 = gen_extra_cond[5:7]

                            return recover_normalized(
                                generate_second_prey_trajectory(
                                    orig_curr_y,
                                    recover_orig_val(curr_prey_1),
                                    recover_orig_val(curr_prey_2), npc_spd_2,
                                    cost_grid, weight_x, weight_y, rot_mat))
                        def f2():
                            return tf.random_uniform([2]) * 2 - 1

                        next_prey_2 = tf.identity(tf.cond(
                            tf.equal(gen_extra_cond[-1], 1), f1, f2), "next")

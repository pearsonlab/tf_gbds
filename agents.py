import tensorflow as tf
# from edward.models import RelaxedOneHotCategorical
from GenerativeModel import HHMM
from RecognitionModel import qHHMM
# from utils import get_vel
# from tf_generate_trial import (recover_orig_val, recover_normalized,
#                                generate_weight, generate_rotation_mat,
#                                generate_prey_trajectory,
#                                generate_second_prey_trajectory,
#                                generate_predator_trajectory)


class game_model(object):
    """Auxiliary class to construct the computational graph
    (define generative and recognition models, draw samples, trial completion)
    """
    def __init__(self, params, inputs, max_vel, n_samples=50):
        with tf.name_scope(params["game_name"]):
            traj = inputs["trajectory"]
            states = inputs["states"]
            ctrl_obs = inputs["observed_control"]
            extra_conds = inputs["extra_conditions"]

            model_dim = params["agent_dim"]
            state_dim = params["state_dim"]
            extra_dim = params["extra_dim"]
            K_1 = params["K_1"]
            K_2 = params["K_2"]

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

                value_shape = [B, Tt, model_dim + K_1 + K_2 + 2]

            self.p = HHMM(params, states, ctrl_obs, extra_conds,
                          name="generative", value=tf.zeros(value_shape))
            self.var_list += self.p.var_list
            self.log_vars += self.p.log_vars

            self.q = qHHMM(params, traj, extra_conds, name="recognition")
            self.var_list += self.q.var_list
            self.log_vars += self.q.log_vars

            self.latent_vars = {self.p: self.q}

            qg_samples = tf.identity(
                self.q.qg.sample(n_samples), "qg_samples")
            qz_1_samples = tf.identity(
                self.q.qz_1.sample(n_samples), "qz_1_samples")
            qz_2_samples = tf.identity(
                self.q.qz_2.sample(n_samples), "qz_2_samples")
            q_exit_samples = tf.identity(
                self.q.exit.sample(n_samples), "q_exit_samples")

            # with tf.name_scope("update_one_step"):
            #     prev_y = tf.placeholder(
            #         tf.float32, obs_dim, "previous_position")
            #     curr_y = tf.placeholder(
            #         tf.float32, obs_dim, "current_position")
            #     v = tf.divide(curr_y - prev_y, max_vel,
            #                   "current_velocity")
            #     curr_s = tf.concat([curr_y, v], 0, "current_state")

            #     if inputs["extra_conds"] is not None:
            #         gen_extra_conds = tf.placeholder(
            #             tf.float32, extra_dim, "extra_conditions")
            #     else:
            #         gen_extra_conds = None

            #     with tf.name_scope("goal"):
            #         prev_g = tf.placeholder(
            #             tf.float32, model_dim, "previous")
            #         curr_g = tf.identity(self.p.update_goal(
            #             curr_s, prev_g, gen_extra_conds), "current")

            #     with tf.name_scope("control"):
            #         with tf.name_scope("error"):
            #             curr_error = tf.subtract(
            #                 curr_g, curr_y, "current")
            #             prev_error = tf.placeholder(
            #                 tf.float32, model_dim, "previous")
            #             prev2_error = tf.placeholder(
            #                 tf.float32, model_dim, "previous2")
            #             errors = tf.stack(
            #                 [prev2_error, prev_error, curr_error], 0,
            #                 "all")
            #         prev_u = tf.placeholder(
            #             tf.float32, model_dim, "previous")
            #         curr_u = tf.identity(self.p.update_ctrl(
            #             errors, prev_u), "current")

            #     if latent_u:
            #         next_y = tf.clip_by_value(
            #             curr_y + max_vel * tf.clip_by_value(
            #                 curr_u, clip_range[:, 0], clip_range[:, 1],
            #                 "clip_control"), -1., 1., "next_position")
            #     else:
            #         next_y = tf.clip_by_value(
            #             curr_y + max_vel * tf.tanh(curr_u), -1., 1.,
            #             "next_position")

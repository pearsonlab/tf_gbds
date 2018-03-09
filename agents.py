import tensorflow as tf
from tf_gbds.GenerativeModel import joint_goals, joint_ctrls
from tf_gbds.RecognitionModel import SmoothingPastLDSTimeSeries
from tf_gbds.utils import Point_Mass


class game_model(object):
    def __init__(self, params, inputs, max_vel, extra_dim=0, n_samples=50):
        with tf.name_scope(params["name"]):
            self.name = params["name"]
            self.obs_dim = params["obs_dim"]

            self.states = inputs["states"]
            self.traj = inputs["trajectories"]
            self.extra_conds = inputs["extra_conds"]
            self.ctrl_obs = inputs["ctrl_obs"]

            self.latent_vars = {}
            self.var_list = []

            with tf.name_scope("variable_value_shape"):
                traj_shape = self.traj.shape.as_list()
                if traj_shape[0] is None:
                    B = 1
                else:
                    B = traj_shape[0]
                if traj_shape[1] is None:
                    Tt = 2
                else:
                    Tt = traj_shape[1]
                value_shape = [B, Tt, self.obs_dim]

            with tf.name_scope("goal"):
                self.g_p = joint_goals(
                    params["agent_priors"], self.states, self.extra_conds,
                    "prior", tf.zeros(value_shape))
                self.var_list += self.g_p.params
                self.g_q = SmoothingPastLDSTimeSeries(
                    params["g_q_params"], self.traj, self.obs_dim,
                    self.obs_dim, self.extra_conds, name="recognition")
                self.var_list += self.g_q.params
                self.latent_vars.update({self.g_p: self.g_q})

                with tf.name_scope("initial_goal"):
                    self.g0 = tf.identity(self.g_p.sample_g0(), "g0")
                    self.g0_samp = tf.identity(self.g_p.sample_g0(1000),
                                               "samples")
                with tf.name_scope("posterior"):
                    self.g_q_mu = tf.identity(
                        tf.squeeze(self.g_q.postX, -1), "mean")
                    self.g_q_samp = tf.identity(
                        self.g_q.sample(n_samples), "samples")

            with tf.name_scope("control"):
                self.u_p = joint_ctrls(
                    params["agent_priors"], self.g_q.value(), self.traj,
                    self.ctrl_obs, "prior", tf.zeros(value_shape))
                if params["u_q_params"] is not None:
                    self.u_q = SmoothingPastLDSTimeSeries(
                        params["u_q_params"], self.traj, self.obs_dim,
                        self.obs_dim, self.extra_conds, name="recognition")
                    self.var_list += self.u_q.params

                    with tf.name_scope("posterior"):
                        self.u_q_mu = tf.identity(
                            tf.squeeze(self.u_q.postX, -1), "mean")
                        self.u_q_samp = tf.identity(
                            self.u_q.sample(n_samples), "samples")
                else:
                    self.u_q = Point_Mass(tf.pad(
                        self.ctrl_obs, [[0, 0], [0, 1], [0, 0]]),
                        name="posterior")
                self.latent_vars.update({self.u_p: self.u_q})

            with tf.name_scope("PID"):
                self.PID_p = params["PID_p"]
                self.PID_q = params["PID_q"]
                self.latent_vars.update({self.PID_p["Kp"]: self.PID_q["Kp"],
                                         self.PID_p["Ki"]: self.PID_q["Ki"],
                                         self.PID_p["Kd"]: self.PID_q["Kd"]})
                self.var_list += self.PID_q["vars"]

            with tf.name_scope("update_one_step"):
                prev_y = tf.placeholder(tf.float32, self.obs_dim,
                                        "previous_position")
                curr_y = tf.placeholder(tf.float32, self.obs_dim,
                                        "current_position")
                v = tf.divide(curr_y - prev_y, max_vel, "current_velocity")
                curr_s = tf.concat([curr_y, v], 0, "current_state")

                if inputs["extra_conds"] is not None:
                    gen_extra_conds = tf.placeholder(
                        tf.float32, extra_dim, "extra_conditions")
                else:
                    gen_extra_conds = None

                with tf.name_scope("goal"):
                    prev_g = tf.placeholder(tf.float32, self.obs_dim,
                                            "previous")
                    curr_g = tf.identity(
                        self.g_p.update_goal(curr_s, prev_g, gen_extra_conds),
                        "current")

                with tf.name_scope("control"):
                    with tf.name_scope("error"):
                        curr_error = tf.subtract(curr_g, curr_y, "current")
                        prev_error = tf.placeholder(tf.float32, self.obs_dim,
                                                    "previous")
                        prev2_error = tf.placeholder(tf.float32, self.obs_dim,
                                                     "previous2")
                        errors = tf.stack(
                            [curr_error, prev_error, prev2_error], 0, "all")
                    prev_u = tf.placeholder(tf.float32, self.obs_dim,
                                            "previous")
                    curr_u = tf.identity(self.u_p.update_ctrl(errors, prev_u),
                                         "current")

                next_y = tf.clip_by_value(
                    curr_y + max_vel * tf.clip_by_value(curr_u, -1., 1.),
                    -1., 1., name="next_position")

            # with tf.name_scope("loss"):
            #     logdensity_g = tf.identity(
            #         self.g_p.log_prob(self.g_q.value()), "goal")
            #     logdensity_u = tf.identity(
            #         self.u_p.log_prob(self.u_q.value()), "control")

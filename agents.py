import tensorflow as tf
from tf_gbds.GenerativeModel import joint_GBDS
from tf_gbds.RecognitionModel import SmoothingPastLDSTimeSeries
from tf_gbds.utils import get_vel, pad_extra_conds


class game_model(object):
    def __init__(self, params, inputs, max_vel, extra_dim=0, n_samples=50):
        with tf.name_scope(params["name"]):
            self.name = params["name"]
            self.obs_dim = params["obs_dim"]

            self.traj = inputs["trajectory"]
            self.states = inputs["states"]
            self.ctrl_obs = inputs["ctrl_obs"]
            self.extra_conds = inputs["extra_conds"]

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
                    Tt = traj_shape[1] - 1
                value_shape = [B, Tt, self.obs_dim]

            self.p = joint_GBDS(
                params["agent_priors"], self.states, self.ctrl_obs,
                self.extra_conds, name="prior",
                value=tf.zeros(value_shape))
            self.var_list += self.p.params
            if isinstance(params["sigma"], tf.Variable):
                self.var_list += [params["sigma"]]
            if isinstance(params["eps"], tf.Variable):
                self.var_list += [params["eps"]]

            self.g_q = SmoothingPastLDSTimeSeries(
                params["g_q_params"], self.traj[:, 1:], self.obs_dim,
                self.obs_dim, self.extra_conds, name="recognition")
            self.var_list += self.g_q.params
            self.latent_vars.update({self.p: self.g_q})

            with tf.name_scope("initial_goal"):
                self.g0 = tf.identity(self.p.sample_g0(), "g0")
                self.g0_samp = tf.identity(self.p.sample_g0(1000),
                                           "samples")

            with tf.name_scope("GMM"):
                traj_i = tf.placeholder(
                    tf.float32, [None, None, self.obs_dim], "trajectory")
                states = get_vel(traj_i, max_vel)
                if extra_dim != 0:
                    extra_conds_i = tf.placeholder(tf.float32, extra_dim,
                                                   "extra_conditions")
                    states = pad_extra_conds(states, extra_conds_i)
                with tf.name_scope("goalie"):
                    goalie = self.p.agents[0]
                    goalie_NN_out = self.p.agents[0].GMM_NN(states[:, 1:])
                    goalie_mu = tf.reshape(
                        goalie_NN_out[:, :, :(goalie.K * goalie.dim)],
                        [tf.shape(traj_i)[0], -1, goalie.K, goalie.dim],
                        "mu")
                    goalie_lambda = tf.reshape(
                        tf.nn.softplus(goalie_NN_out[:, :, (
                            goalie.K * goalie.dim):(
                                2 * goalie.K * goalie.dim)]),
                        [tf.shape(traj_i)[0], -1, goalie.K, goalie.dim],
                        "lambda")
                    goalie_w = tf.nn.softmax(tf.reshape(
                        goalie_NN_out[:, :, (2 * goalie.K * goalie.dim):],
                        [tf.shape(traj_i)[0], -1, goalie.K]), -1, "w")
                with tf.name_scope("shooter"):
                    shooter = self.p.agents[1]
                    shooter_NN_out = self.p.agents[1].GMM_NN(states[:, 1:])
                    shooter_mu = tf.reshape(
                        shooter_NN_out[:, :, :(shooter.K * shooter.dim)],
                        [tf.shape(traj_i)[0], -1, shooter.K, shooter.dim],
                        "mu")
                    shooter_lambda = tf.reshape(
                        tf.nn.softplus(shooter_NN_out[:, :, (
                            shooter.K * shooter.dim):(
                                2 * shooter.K * shooter.dim)]),
                        [tf.shape(traj_i)[0], -1, shooter.K, shooter.dim],
                        "lambda")
                    shooter_w = tf.nn.softmax(tf.reshape(
                        shooter_NN_out[:, :, (2 * shooter.K * shooter.dim):],
                        [tf.shape(traj_i)[0], -1, shooter.K]), -1, "w")

            with tf.name_scope("posterior"):
                self.g_q_mu = tf.identity(
                    tf.squeeze(self.g_q.postX, -1), "mean")
                self.g_q_samp = tf.identity(
                    self.g_q.sample(n_samples), "samples")

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
                        self.p.update_goal(curr_s, prev_g, gen_extra_conds),
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
                    curr_u = tf.identity(self.p.update_ctrl(errors, prev_u),
                                         "current")

                next_y = tf.clip_by_value(
                    curr_y + max_vel * tf.tanh(curr_u), -1., 1.,
                    name="next_position")

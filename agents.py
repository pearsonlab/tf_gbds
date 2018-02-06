import tensorflow as tf
from tf_gbds.GenerativeModel import GBDS_g, GBDS_u, joint_goals, joint_ctrls
from tf_gbds.RecognitionModel import SmoothingPastLDSTimeSeries
from tf_gbds.utils import generate_trial


class agent_model(object):
    def __init__(self, params, inputs):
        with tf.name_scope(params["name"]):
            self.name = params["name"]
            self.dim = params["dim"]
            self.col = tf.identity(params["col"],
                                   name="agent_columns")
            self.obs_dim = params["obs_dim"]

            self.states = tf.identity(inputs["states"], name="states")
            self.traj = tf.identity(inputs["trajectories"],
                                    name="trajectories")
            if "extra_conds" in inputs:
                self.extra_conds = tf.identity(inputs["extra_conds"],
                                               name="extra_conditions")
            else:
                self.extra_conds = None
            if "ctrl_obs" in inputs:
                self.ctrl_obs = tf.identity(inputs["ctrl_obs"],
                                            name="observed_control")
            else:
                self.ctrl_obs = None

            self.vars = []
            value_shape = [1, 2, self.dim]
            latent_u = params["latent_u"]

            with tf.name_scope("goal"):
                self.g_p = GBDS_g(params, self.states, self.extra_conds,
                                  name="prior", value=tf.zeros(value_shape))
                self.vars += self.g_p.params
                self.g_q = SmoothingPastLDSTimeSeries(
                    params["g_q_params"], self.traj, self.dim, self.obs_dim,
                    self.extra_conds, name="posterior")
                self.vars += self.g_q.params
                self.goal = {self.g_p: self.g_q}

            with tf.name_scope("control"):
                self.u_p = GBDS_u(
                    params, self.g_q.value(),
                    tf.gather(self.traj, self.col, axis=-1), self.ctrl_obs,
                    name="prior", value=tf.zeros(value_shape))
                if latent_u:
                    self.u_q = SmoothingPastLDSTimeSeries(
                        params["u_q_params"], self.traj, self.dim,
                        self.obs_dim, self.extra_conds, name="posterior")
                    self.vars += self.u_q.params
                else:
                    self.u_q = tf.pad(
                        self.u_p.ctrl_obs, [[0, 0], [0, 1], [0, 0]],
                        name="posterior")
                self.ctrl = {self.u_p: self.u_q}

            with tf.name_scope("PID"):
                self.PID_p = params["PID_p"]
                self.PID_q = params["PID_q"]
                self.PID = {self.PID_p["Kp"]: self.PID_q["Kp"],
                            self.PID_p["Ki"]: self.PID_q["Ki"],
                            self.PID_p["Kd"]: self.PID_q["Kd"]}
                self.vars += self.PID_q["vars"]

        super(agent_model, self).__init__()


class game_model(object):
    def __init__(self, params, inputs, n_samples=50, name="penaltykick"):
        with tf.name_scope(name):
            self.name = name
            self.latent_vars = {}
            self.data = {}
            self.var_list = []

            if isinstance(params, list):
                self.agents = [agent_model(p, inputs)
                               for p in params]
            else:
                raise TypeError("params must be a list object.")

            for agent in self.agents:
                self.latent_vars.update(agent.goal)
                self.data.update(agent.ctrl)
                self.latent_vars.update(agent.PID)
                self.var_list += agent.vars

            self.g = joint_goals([agent.g_p for agent in self.agents],
                                 [agent.g_q for agent in self.agents],
                                 name="goals")
            self.g0_samp = tf.identity(self.g.sample_g0(n=n_samples),
                                       name="initial_goal_samples")
            self.g_q_mu = tf.identity(
                self.g.q_mean, name="goal_posterior_mean")
            self.g_q_samp = tf.identity(
                self.g.sample_posterior(n=n_samples),
                name="goal_posterior_samples")

            self.u = joint_ctrls([agent.u_p for agent in self.agents],
                                 [agent.u_q for agent in self.agents],
                                 name="controls")
            if hasattr(self.u, "q_mean"):
                self.u_q_mu = tf.identity(self.u.q_mean,
                                          name="control_posterior_mean")
                self.u_q_samp = tf.identity(
                    self.u.sample_posterior(n=n_samples),
                    name="control_posterior_samples")

        super(game_model, self).__init__()

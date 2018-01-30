import tensorflow as tf
from tf_gbds.GenerativeModel import GBDS_g, GBDS_u, joint_goals, joint_ctrls
from tf_gbds.RecognitionModel import SmoothingPastLDSTimeSeries
from tf_gbds.utils import get_PID_posteriors, get_rec_params, get_vel


class agent_model(object):
    def __init__(self, params, states, trajectories, hps):
        with tf.name_scope(params['name']):
            self.dim = params['agent_dim']
            self.cols = tf.identity(params['agent_cols'],
                                    name='agent_columns')
            self.states = tf.identity(states, name='states')
            self.traj = tf.identity(trajectories, name='trajectories')
            self.obs_dim = self.traj.shape[-1].value

            self.name = params['name']
            self.vars = []
            value_shape = [1, 2, self.dim]

            with tf.name_scope('goal'):
                self.g_p = GBDS_g(params, self.states, name='prior',
                                  value=tf.zeros(value_shape))
                self.vars += self.g_p.params
                g_q_params = get_rec_params(
                    self.obs_dim, hps.extra_dim, self.dim, hps.lag,
                    hps.n_layers_rec, hps.hidden_dim_rec, name='posterior')
                self.g_q = SmoothingPastLDSTimeSeries(
                    g_q_params, self.traj, self.obs_dim, self.dim,
                    name='posterior')
                self.vars += self.g_q.params
                self.goal = {self.g_p: self.g_q}

            with tf.name_scope('ctrl'):
                self.u_p = GBDS_u(
                    params, self.g_q.value(),
                    tf.gather(self.traj, self.cols, axis=-1),
                    name='prior', value=tf.zeros(value_shape))
                if hps.latent_u:
                    u_q_params = get_rec_params(
                        self.obs_dim, hps.extra_dim, self.dim, hps.lag,
                        hps.n_layers_rec, hps.hidden_dim_rec,
                        name='posterior')
                    self.u_q = SmoothingPastLDSTimeSeries(
                        u_q_params, self.traj, self.obs_dim, self.dim,
                        name='posterior')
                    self.vars += self.u_q.params
                else:
                    self.u_q = tf.pad(
                        self.u_p.ctrl_obs, [[0, 0], [0, 1], [0, 0]],
                        name='posterior')
                self.ctrl = {self.u_p: self.u_q}

            with tf.name_scope('PID'):
                self.PID_priors = params['PID_params']
                self.PID_posteriors = get_PID_posteriors(self.dim)
                self.PID = {self.PID_priors['Kp']: self.PID_posteriors['Kp'],
                            self.PID_priors['Ki']: self.PID_posteriors['Ki'],
                            self.PID_priors['Kd']: self.PID_posteriors['Kd']}
                self.vars += self.PID_posteriors['vars']

        super(agent_model, self).__init__()


class game_model(object):
    def __init__(self, agents, inputs, hps, name='penaltykick'):

        with tf.name_scope('get_states'):
            self.traj = inputs['trajectories']
            self.states = get_vel(self.traj, hps.max_vel)

        self.name = name
        self.latent_vars = {}
        self.data = {}
        self.var_list = []

        with tf.name_scope('agents'):
            if isinstance(agents, list):
                self.agents = [agent_model(
                    agent['params'], self.states, self.traj, hps)
                           for agent in agents]
            else:
                raise TypeError('agents must be a list object.')

        for agent in self.agents:
            self.latent_vars.update(agent.goal)
            self.data.update(agent.ctrl)
            self.latent_vars.update(agent.PID)
            self.var_list += agent.vars

        super(game_model, self).__init__()

#   def generate_trials(self, n, y0):

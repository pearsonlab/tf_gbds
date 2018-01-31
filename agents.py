import tensorflow as tf
from tf_gbds.GenerativeModel import GBDS_g, GBDS_u, joint_goals, joint_ctrls
from tf_gbds.RecognitionModel import SmoothingPastLDSTimeSeries
from tf_gbds.utils import generate_trial


class agent_model(object):
    def __init__(self, params, states, trajectories):
        with tf.name_scope(params['name']):
            self.dim = params['agent_dim']
            self.cols = tf.identity(params['agent_cols'],
                                    name='agent_columns')
            self.states = tf.identity(states, name='states')
            self.traj = tf.identity(trajectories, name='trajectories')
            self.obs_dim = params['obs_dim']

            self.name = params['name']
            self.vars = []
            value_shape = [1, 2, self.dim]
            latent_u = params['latent_u']

            with tf.name_scope('goal'):
                self.g_p = GBDS_g(params, self.states, name='prior',
                                  value=tf.zeros(value_shape))
                self.vars += self.g_p.params
                self.g_q = SmoothingPastLDSTimeSeries(
                    params['g_q_params'], self.traj, self.obs_dim, self.dim,
                    name='posterior')
                self.vars += self.g_q.params
                self.goal = {self.g_p: self.g_q}

            with tf.name_scope('control'):
                self.u_p = GBDS_u(params, self.g_q.value(),
                                  tf.gather(self.traj, self.cols, axis=-1),
                                  name='prior', value=tf.zeros(value_shape))
                if latent_u:
                    self.u_q = SmoothingPastLDSTimeSeries(
                        params['u_q_params'], self.traj, self.obs_dim,
                        self.dim, name='posterior')
                    self.vars += self.u_q.params
                else:
                    self.u_q = tf.pad(
                        self.u_p.ctrl_obs, [[0, 0], [0, 1], [0, 0]],
                        name='posterior')
                self.ctrl = {self.u_p: self.u_q}

            with tf.name_scope('PID'):
                self.PID_p = params['PID_priors']
                self.PID_q = params['PID_posteriors']
                self.PID = {self.PID_p['Kp']: self.PID_q['Kp'],
                            self.PID_p['Ki']: self.PID_q['Ki'],
                            self.PID_p['Kd']: self.PID_q['Kd']}
                self.vars += self.PID_q['vars']

        super(agent_model, self).__init__()


class game_model(object):
    def __init__(self, agents, inputs, name='penaltykick'):

        self.states = inputs['states']
        self.traj = inputs['trajectories']

        self.name = name
        self.latent_vars = {}
        self.data = {}
        self.var_list = []

        with tf.name_scope('model'):
            if isinstance(agents, list):
                self.agents = [agent_model(
                    agent['params'], self.states, self.traj)
                               for agent in agents]
            else:
                raise TypeError('agents must be a list object.')

        for agent in self.agents:
            self.latent_vars.update(agent.goal)
            self.data.update(agent.ctrl)
            self.latent_vars.update(agent.PID)
            self.var_list += agent.vars

        self.g = joint_goals([agent.g_p for agent in self.agents],
                             [agent.g_q for agent in self.agents],
                             name='goals')
        self.g0_samp = self.g.sample_g0(n=100)
        self.g_q_samp = self.g.sample_posterior(n=20)
        self.u = joint_ctrls([agent.u_p for agent in self.agents],
                             [agent.u_q for agent in self.agents],
                             name='controls')
        # self.u_q_samp = self.u.sample_posterior(n=20)

        self.generated_trial, _, _ = generate_trial(self.g, self.u)

        super(game_model, self).__init__()

import tensorflow as tf
import numpy as np
from edward.models import PointMass
from tf_gbds.GenerativeModel import GBDS_g, GBDS_u, joint_goals, joint_ctrls
from tf_gbds.RecognitionModel import SmoothingPastLDSTimeSeries
from tf_gbds.utils import get_PID_posteriors, get_rec_params, get_vel


class agent_model(object):
    def __init__(self, params, states):
        super(agent_model, self).__init__()

        self.dim = params['agent_dim']
        self.cols = params['agent_cols']
        value_shape = [1, 2, self.dim]
        self.states = states
        # extra_conds, ctrl_obs
        self.name = params['name']

        self.goal = GBDS_g(params, self.states, name='%s_goal' % self.name,
                           value=tf.zeros(value_shape))
        self.ctrl = GBDS_u(
            params, self.goal, tf.gather(self.states, self.cols, axis=-1),
            name='%s_ctrl' % self.name, value=tf.zeros(value_shape))

        PID_priors = params['PID_params']
        PID_posteriors = get_PID_posteriors(self.name, self.dim)
        self.PID = {PID_priors['Kp']: PID_posteriors['Kp'],
                    PID_priors['Ki']: PID_posteriors['Ki'],
                    PID_priors['Kd']: PID_posteriors['Kd']}

        self.vars = self.goal.getParams() + PID_posteriors['vars']


class game_model(object):
    def __init__(self, agents, inputs, hps, name='penaltykick'):
        super(game_model, self).__init__()

        self.trajectories = inputs['trajectories']
        self.obs_dim = tf.shape(self.trajectories)[-1]
        self.states = get_vel(self.trajectories, hps.max_vel)
        self.model_dim = np.sum(agent['params']['agent_dim']
                                for agent in agents)
        self.name = name
        value_shape = [1, 2, self.model_dim]
        self.latent_vars = {}
        self.var_list = []

        if isinstance(agents, list):
            self.agents = [agent_model(agent['params'], self.states)
                           for agent in agents]
        else:
            raise TypeError('agents must be a list object.')

        for agent in self.agents:
            self.latent_vars.update(agent.PID)
            self.var_list += agent.vars

        goals = [agent.goal for agent in self.agents]
        self.joint_goals_prior = joint_goals(
            goals, name='%s_goals_prior' % name, value=tf.zeros(value_shape))

        goals_posterior_params = get_rec_params(
            self.model_dim, hps.extra_dim, hps.lag, hps.n_layers_rec,
            hps.hidden_dim_rec, name='%s_goals_posterior' % name)
        self.joint_goals_posterior = SmoothingPastLDSTimeSeries(
            goals_posterior_params, self.trajectories, self.obs_dim,
            self.model_dim, name='%s_goals_posterior' % name)

        self.latent_vars.update(
            {self.joint_goals_prior: self.joint_goals_posterior})
        self.var_list += self.joint_goals_posterior.getParams()

        ctrls = [agent.ctrl for agent in self.agents]
        self.joint_ctrls_prior = joint_ctrls(
        	ctrls, name='%s_ctrls_prior' % name, value=tf.zeros(value_shape))

        if hps.latent_u:
            ctrls_posterior_params = get_rec_params(
                self.model_dim, hps.extra_dim, hps.lag, hps.n_layers_rec,
                hps.hidden_dim_rec, name='%s_ctrls_posterior' % name)
            self.joint_ctrls_posterior = SmoothingPastLDSTimeSeries(
                ctrls_posterior_params, self.trajectories, self.obs_dim,
                self.model_dim, name='%s_ctrls_posterior' % name)
            self.var_list += self.joint_ctrls_posterior.getParams()
        else:
            self.joint_ctrls_posterior = self.joint_ctrls_prior.ctrl_obs

        self.data = {self.joint_ctrls_prior: self.joint_ctrls_posterior}

#   def generate_trials(self, n, y0):

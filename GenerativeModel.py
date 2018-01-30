import tensorflow as tf
import numpy as np
from edward.models import RandomVariable
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED)
from tensorflow.python.ops.distributions.special_math import log_ndtr
from tf_gbds.utils import pad_extra_conds


class GBDS_u(RandomVariable, Distribution):
    """A customized Random Variable of control signal in Goal-Based Dynamical
       System
    """
    def __init__(self, GenerativeParams, goals, positions, ctrl_obs=None,
                 name='GBDS_u', value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):

        """Initialize a batch of GBDS_u random variables for one of the agents

        Args:
          GenerativeParams: A dictionary of parameters for the agent
          Entries include:
            - get_states: function that calculates current state from position
            - pen_eps: Penalty on control signal noise, epsilon
            - pen_sigma: Penalty on goals state noise, sigma
            - pen_g: Two penalties on goal state leaving boundaries (Can be set
                     None)
            - bounds_g: Boundaries corresponding to above penalties
            - NN_postJ_mu: Neural network that parametrizes the mean of the
                           posterior of J (i.e. mu and sigma), conditioned on
                           goals
            - NN_postJ_sigma: Neural network that parametrizes the covariance
                              of the posterior of J (i.e. mu and sigma),
                              conditioned on goals
            - yCols: Columns of Y this agent corresponds to. Used to index
                     columns in real data to compare against generated data.
            - vel: Maximum velocity of each dimension in columns belong to this
                   agent(yCols)
            - all_vel: Maximum velocity of each dimension in columns belong to
                       all agents
            - clip: Clipping signal
            - clip_range: the range of the clipping states
            - clip_tol: the tolenrance of clipping
            - GMM_net: Gaussian Mixture Model network
            - GMM_k: Number of GMM components
          g: The dependent customized Random Variable of goal in Goal-Based
             Dynamical System (GBDS_g)
          y: Time series of positions
          yDim: Number of dimensions for this agent
          value: The Random Variable sample of control signal
        """

        with tf.name_scope(name):
            self.g = tf.identity(goals, name='goals')
            self.y = tf.identity(positions, name='positions')
            self.dim = GenerativeParams['agent_dim']
            self.cols = tf.identity(GenerativeParams['agent_cols'],
                                    name='agent_columns')
            self.B = tf.identity(tf.shape(positions)[0], name='batch_size')
            self.Tt = tf.identity(tf.shape(positions)[1], name='trial_length')
            self.vel = tf.constant(
                GenerativeParams['agent_vel'], dtype=tf.float32,
                shape=[self.dim], name='velocity')
            self.pen_u = tf.constant(
                GenerativeParams['pen_u'], dtype=tf.float32,
                name='control_residual_penalty')
            self.res_tol = tf.constant(
                GenerativeParams['u_res_tol'], dtype=tf.float32,
                name='control_residual_tolerance')

            with tf.name_scope('PID_control'):
                with tf.name_scope('parameters'):
                    # coefficients for PID controller (one for each dimension)
                    # https://en.wikipedia.org/wiki/PID_controller
                    # Discrete_implementation
                    PID_params = GenerativeParams['PID_params']
                    # priors of PID parameters
                    self.Kp = PID_params['Kp']
                    self.Ki = PID_params['Ki']
                    self.Kd = PID_params['Kd']
                    self.params = [self.Kp] + [self.Ki] + [self.Kd]
                with tf.name_scope('filter'):
                    # calculate coefficients in convolutional filter
                    t_coeff = self.Kp + self.Ki + self.Kd
                    t1_coeff = -self.Kp - 2 * self.Kd
                    t2_coeff = self.Kd
                    # concatenate coefficients into filter
                    self.L = tf.stack([t2_coeff, t1_coeff, t_coeff], axis=1,
                                      name='PID_convolution_filter')

            with tf.name_scope('control_signal'):
                if ctrl_obs is not None:
                    self.ctrl_obs = tf.identity(ctrl_obs,
                                                name='observed_control')
                else:
                    self.ctrl_obs = tf.divide(
                        tf.subtract(self.y[:, 1:], self.y[:, :-1],
                                    name='distance'),
                        self.vel, name='observed_control')
            with tf.name_scope('control_signal_censoring'):
                # clipping signal
                self.clip = GenerativeParams['clip']
                if self.clip:
                    self.clip_range = GenerativeParams['clip_range']
                    self.clip_tol = GenerativeParams['clip_tol']
                    self.eta = GenerativeParams['eta']
            # with tf.name_scope('control_signal_noise'):
            #     # noise coefficient on control signals
            #     self.eps = tf.constant(
            #         GenerativeParams['epsilon'] * np.ones((1, self.dim)),
            #         dtype=tf.float32, name='epsilon')
            with tf.name_scope('control_signal_penalty'):
                # penalty on control error
                if GenerativeParams['pen_ctrl_error'] is not None:
                    self.pen_ctrl_error = tf.constant(
                        GenerativeParams['pen_ctrl_error'], dtype=tf.float32,
                        name='control_error_penalty')
                else:
                    self.pen_ctrl_error = None

        super(GBDS_u, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs['goals'] = goals
        self._kwargs['positions'] = positions
        self._kwargs['GenerativeParams'] = GenerativeParams

    def get_preds(self, y, post_g, u_prev):
        """Return one-step-ahead prediction of control signal,
        given samples from goal posterior.
        """

        error = tf.subtract(post_g, y, name='control_error')

        u_diff = []
        # get current error signal and corresponding filter
        for i in range(self.dim):
            signal = error[:, :, i]
            # pad the beginning of control signal with zero
            signal = tf.expand_dims(
                tf.pad(signal, [[0, 0], [2, 0]], name='zero_pad'),
                -1, name='reshape_signal')
            filt = tf.reshape(self.L[i], [-1, 1, 1],
                              name='reshape_filter')
            res = tf.nn.convolution(signal, filt, padding='VALID',
                                    name='convolve_signal')
            u_diff.append(res)
        if len(u_diff) > 1:
            u_diff = tf.concat([*u_diff], axis=-1,
                               name='control_signal_change')
        else:
            u_diff = tf.identity(u_diff[0], name='contrl_signal_change')

        u_pred = tf.add(u_prev, u_diff, name='predicted_control_signal')

        return (error, u_pred)

    def clip_log_prob(self, upsilon, u):
        """upsilon (derived from time series of y) is a censored version of
        a noisy control signal: \hat{u} ~ N(u, \eta^2).
        log p(upsilon|u, g) = log p(upsilon|u) + log(u|g)
        log p(upsilon|u) breaks down into three cases,
        namely left-clipped (upsilon_t = -1), right-clipped (upsilon_t = 1),
        and non-clipped (-1 < upsilon_t < 1). For the first two cases,
        Normal CDF is used instead of PDF due to censoring.
        The log density term is computed for each and then add up.
        """

        l_b = tf.add(self.clip_range[0], self.clip_tol, name='lower_bound')
        u_b = tf.subtract(self.clip_range[1], self.clip_tol,
                          name='upper_bound')
        eta = self.eta

        def z(x, loc, scale):
            return (x - loc) / scale

        def normal_logpdf(x, loc, scale):
            return -(0.5 * np.log(2 * np.pi) + tf.log(scale) +
                     0.5 * tf.square(z(x, loc, scale)))

        def normal_logcdf(x, loc, scale):
            return log_ndtr(z(x, loc, scale))

        return tf.where(tf.less_equal(upsilon, l_b, name='left_clip'),
                        normal_logcdf(-1., u, eta),
                        tf.where(tf.greater_equal(upsilon, u_b,
                                                  name='right_clip'),
                                 normal_logcdf(-1., -u, eta),
                                 normal_logpdf(upsilon, u, eta)))

    def _log_prob(self, value):
        """Evaluates the log-density of the GenerativeModel.
        """

        # Get predictions for next timestep (at each timestep except for last)
        # disregard last timestep bc we don't know the next value, thus, we
        # can't calculate the error

        with tf.name_scope('next_step_prediction'):
            ctrl_error, u_pred = self.get_preds(
                self.y[:, :-1], self.g[:, 1:],
                tf.pad(value[:, :-2], [[0, 0], [1, 0], [0, 0]],
                       name='previous_control'))

        LogDensity = 0.0
        # calculate loss on control signal
        with tf.name_scope('clipping_noise'):
            if self.clip:
                LogDensity += tf.reduce_sum(
                    self.clip_log_prob(self.ctrl_obs, value[:, :-1]), [1, 2])

        with tf.name_scope('control_signal'):
            u_res = tf.subtract(value[:, :-1], u_pred,
                                name='control_signal_residual')
            # LogDensity -= tf.reduce_sum(
            #     (0.5 * tf.log(2 * np.pi) + tf.log(self.eps) +
            #      u_res ** 2 / (2 * self.eps ** 2)), [1, 2])
            LogDensity -= tf.reduce_sum(self.pen_u * tf.nn.relu(
                u_res - self.res_tol), axis=[1, 2])
            LogDensity -= tf.reduce_sum(self.pen_u * tf.nn.relu(
                -u_res - self.res_tol), axis=[1, 2])
        with tf.name_scope('control_error_penalty'):
            # penalty on ctrl error
            if self.pen_ctrl_error is not None:
                LogDensity -= self.pen_ctrl_error * tf.reduce_sum(
                    tf.nn.relu(ctrl_error - 1.0), [1, 2])
                LogDensity -= self.pen_ctrl_error * tf.reduce_sum(
                    tf.nn.relu(-ctrl_error - 1.0), [1, 2])

        return tf.reduce_mean(LogDensity) / tf.cast(self.Tt - 1, tf.float32)

    def update_ctrl(self, errors, u_prev):
        u_diff = tf.reduce_sum(errors * tf.transpose(self.L), 0,
                               name='control_signal_change')
        u = tf.add(u_prev, u_diff, name='update_control')

        return u


class GBDS_g(RandomVariable, Distribution):
    """A customized Random Variable of goal in Goal-Based Dynamical System
    """

    def __init__(self, GenerativeParams, states, extra_conds=None,
                 name='GBDS_g', value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):
        """Initialize a batch of GBDS_g random variables for one of the agent


        Args:
          GenerativeParams: A dictionary of parameters for the agent
          Entries include:
            - get_states: function that calculates current state from position
            - pen_eps: Penalty on control signal noise, epsilon
            - pen_sigma: Penalty on goals state noise, sigma
            - pen_g: Two penalties on goal state leaving boundaries (Can be set
                     None)
            - bounds_g: Boundaries corresponding to above penalties
            - NN_postJ_mu: Neural network that parametrizes the mean of the
                           posterior of J (i.e. mu and sigma), conditioned on
                           goals
            - NN_postJ_sigma: Neural network that parametrizes the covariance
                              of the posterior of J (i.e. mu and sigma),
                              conditioned on goals
            - yCols: Columns of Y this agent corresponds to. Used to index
                     columns in real data to compare against generated data.
            - vel: Maximum velocity of each dimension in columns belong to this
                   agent(yCols)
            - all_vel: Maximum velocity of each dimension in columns belong to
                       all agents
            - clip: Clipping signal
            - clip_range: the range of the clipping states
            - clip_tol: the tolenrance of clipping
            - GMM_net: Gaussian Mixture Model network
            - GMM_k: Number of GMM components
          yDim: Number of dimensions for the agent
          y: Time series of positions
          value: The Random Variable sample of goal.
        """

        with tf.name_scope(name):
            self.s = tf.identity(states, name='states')
            self.dim = GenerativeParams['agent_dim']
            self.cols = tf.identity(GenerativeParams['agent_cols'],
                                    name='agent_columns')
            self.B = tf.identity(tf.shape(states)[0], name='batch_size')
            self.Tt = tf.identity(tf.shape(states)[1], name='trial_length')

            with tf.name_scope('extra_conditions'):
                if extra_conds is not None:
                    self.extra_conds = tf.identity(extra_conds,
                                                   name='extra_conditions')
                else:
                    self.extra_conds = None

            with tf.name_scope('g0'):
                g0_params = GenerativeParams['g0_params']
                self.g0_mu = tf.identity(g0_params['mu'], name='mu')
                self.g0_lambda = tf.nn.softplus(
                    g0_params['unc_lambda'], name='lambda')
                self.g0_w = tf.nn.softmax(g0_params['unc_w'], name='w')
                self.g0_params = ([g0_params['mu']] +
                                  [g0_params['unc_lambda']] +
                                  [g0_params['unc_w']])

            with tf.name_scope('GMM_neural_network'):
                # number of GMM components
                self.GMM_K = GenerativeParams['GMM_K']
                self.GMM_NN = GenerativeParams['GMM_NN']

            self.params = self.g0_params + self.GMM_NN.variables

            with tf.name_scope('goal_state_noise'):
                # noise coefficient on goal states
                self.sigma = tf.constant(
                    GenerativeParams['sigma'] * np.ones((1, self.dim)),
                    dtype=tf.float32, name='sigma')

            with tf.name_scope('goal_state_boundary'):
                # corresponding boundaries for pen_g
                if GenerativeParams['bounds_g'] is not None:
                    self.bounds_g = GenerativeParams['bounds_g']
                else:
                    self.bounds_g = [-1., 1.]
            with tf.name_scope('goal_state_penalty'):
                # penalty on goal state escaping boundaries
                if GenerativeParams['pen_g'] is not None:
                    self.pen_g = tf.constant(
                        GenerativeParams['pen_g'], dtype=tf.float32,
                        name='goal_boundary_penalty')
                else:
                    self.pen_g = None

        super(GBDS_g, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs['states'] = states
        self._kwargs['GenerativeParams'] = GenerativeParams

    def get_preds(self, s, g, extra_conds=None):
        """Return one-step-ahead prediction of goals, given states.
        """

        with tf.name_scope('pad_extra_conds'):
            if extra_conds is not None:
                s = pad_extra_conds(s, extra_conds)

        all_mu = tf.reshape(
            self.GMM_NN(s)[:, :, :(self.GMM_K * self.dim)],
            [self.B, -1, self.GMM_K, self.dim], name='all_mu')

        all_lambda = tf.reshape(tf.nn.softplus(
            self.GMM_NN(s)[:, :, (self.GMM_K * self.dim):(
                2 * self.GMM_K * self.dim)], name='softplus_lambda'),
            [self.B, -1, self.GMM_K, self.dim], name='all_lambda')

        all_w = tf.nn.softmax(tf.reshape(self.GMM_NN(s)[:, :, (
            2 * self.GMM_K * self.dim):], [self.B, -1, self.GMM_K],
            name='reshape_w'), dim=-1, name='all_w')

        next_g = tf.divide(tf.expand_dims(g, 2) + all_mu * all_lambda,
                           1 + all_lambda, name='next_goals')

        return (all_mu, all_lambda, all_w, next_g)

    def _log_prob(self, value):
        with tf.name_scope('next_step_prediction'):
            _, all_lambda, all_w, g_pred = self.get_preds(
                self.s[:, :-1], value[:, :-1], self.extra_conds)

        LogDensity = 0.0
        with tf.name_scope('goal_states'):
            # w_brdcst = tf.expand_dims(all_w, -1, name='reshape_w')
            res_gmm = tf.subtract(
                tf.expand_dims(value[:, 1:], 2, name='reshape_samples'),
                g_pred, name='GMM_residual')
            gmm_term = tf.log(all_w + 1e-8) - tf.reduce_sum(
                (1 + all_lambda) * (res_gmm ** 2) / (2 * self.sigma ** 2), -1)
            gmm_term += (0.5 * tf.reduce_sum(tf.log(1 + all_lambda), -1) -
                         tf.reduce_sum(0.5 * tf.log(2 * np.pi) +
                                       tf.log(self.sigma), -1))
            LogDensity += tf.reduce_sum(tf.reduce_logsumexp(gmm_term, -1), 1)

        with tf.name_scope('g0'):
            res_g0 = tf.subtract(tf.expand_dims(value[:, 0], 1), self.g0_mu,
                                 name='g0_residual')
            g0_term = tf.log(self.g0_w + 1e-8) - tf.reduce_sum(
                self.g0_lambda * (res_g0 ** 2) / 2, -1)
            g0_term += 0.5 * tf.reduce_sum(
                tf.log(self.g0_lambda) - tf.log(2 * np.pi), -1)
            LogDensity += tf.reduce_logsumexp(g0_term, -1)

        with tf.name_scope('boundary_penalty'):
            if self.pen_g is not None:
                # linear penalty on goal state escaping game space
                LogDensity -= self.pen_g * tf.reduce_sum(
                    tf.nn.relu(self.bounds_g[0] - value), [1, 2])
                LogDensity -= self.pen_g * tf.reduce_sum(
                    tf.nn.relu(value - self.bounds_g[1]), axis=[1, 2])

        return tf.reduce_mean(LogDensity) / tf.cast(self.Tt, tf.float32)

    def sample_g0(self):
        with tf.name_scope('select_component'):
            k0 = tf.squeeze(tf.multinomial(tf.reshape(
                tf.log(self.g0_w, name='log_g0_w'), [1, -1]),
                1, name='draw_sample'), name='k0')

        with tf.name_scope('get_sample'):
            g0 = tf.add((tf.random_normal([self.dim], name='std_normal') /
                         tf.sqrt(self.g0_lambda[k0], name='inv_std_dev')),
                        self.g0_mu[k0], name='g0')

        return g0

    def sample_GMM(self, state, g_prev):
        state = tf.reshape(state, [1, 1, -1], name='reshape_state')
        with tf.name_scope('mu'):
            all_mu = tf.reshape(
                self.GMM_NN(state)[:, :, :(self.GMM_K * self.dim)],
                [self.GMM_K, self.dim], name='all_mu')

        with tf.name_scope('lambda'):
            all_lambda = tf.reshape(tf.nn.softplus(
                self.GMM_NN(state)[:, :, (self.GMM_K * self.dim):(
                    2 * self.GMM_K * self.dim)], name='softplus_lambda'),
                [self.GMM_K, self.dim], name='all_lambda')

        with tf.name_scope('w'):
            all_w = tf.reshape(tf.nn.softmax(
                self.GMM_NN(state)[:, :, (2 * self.GMM_K * self.dim):],
                dim=-1, name='softmax_w'), [1, self.GMM_K], name='all_w')

        with tf.name_scope('select_component'):
            k = tf.squeeze(tf.multinomial(
                tf.reshape(tf.log(all_w, name='log_w'), [1, -1]),
                1, name='draw_sample'), name='k')

        with tf.name_scope('get_sample'):
            g = tf.add(tf.divide(g_prev + all_mu[k] * all_lambda[k],
                                 1 + all_lambda[k], name='mean'),
                       (tf.random_normal([self.dim], name='std_normal') *
                        tf.divide(tf.squeeze(self.sigma),
                                  tf.sqrt(1 + all_lambda[k]), name='std_dev')),
                       name='goal')

        return g


class joint_goals(RandomVariable, Distribution):
    def __init__(self, goals, name='goals', value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):
        if isinstance(goals, list):
            self.goals = goals
            self.n_agents = len(goals)
            self.dim = np.sum(goal.dim for goal in goals)
        else:
            raise TypeError('goals must be a list.')

        super(joint_goals, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)
        self._kwargs['goals'] = goals

    def _log_prob(self, value):
        log_prob_sum = 0.0

        for agent_goal in self.goals:
            agent_value = tf.gather(value, agent_goal.cols, axis=-1)
            log_prob_sum += agent_goal.log_prob(agent_value)

        return log_prob_sum
        # return tf.reduce_sum(goal.log_prob(
        #     tf.gather(value, goal.cols, axis=-1)) for goal in self.goals)

    def sample_g0(self):
        return tf.concat([goal.sample_g0() for goal in self.goals], 0,
                         name='concat_g0')

    def sample_GMM(self, state, g_prev):
        return tf.concat(
            [goal.sample_GMM(state, tf.gather(g_prev, goal.cols, axis=-1))
             for goal in self.goals], 0, name='concat_g')

    # def update_ctrl(self, errors, u_prev):
    #     return tf.concat(
    #         [goal.update_ctrl(tf.gather(errors, goal.cols, axis=-1),
    #                           tf.gather(u_prev, goal.cols, axis=-1))
    #                      for goal in self.goals], 0, name='concat_u')


class joint_ctrls(RandomVariable, Distribution):
    def __init__(self, ctrls, name='ctrls', value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):
        if isinstance(ctrls, list):
            self.ctrls = ctrls
            self.n_agents = len(ctrls)
            self.dim = np.sum(ctrl.dim for ctrl in ctrls)
            self.ctrl_obs = tf.pad(
                tf.concat([ctrl.ctrl_obs for ctrl in ctrls],
                          -1, name='concat_ctrl_obs'),
                [[0, 0], [0, 1], [0, 0]], name='ctrl_obs')
        else:
            raise TypeError('ctrls must be a list.')

        super(joint_ctrls, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)
        self._kwargs['ctrls'] = ctrls

    def _log_prob(self, value):
        log_prob_sum = 0.0

        for agent_ctrl in self.ctrls:
            agent_value = tf.gather(value, agent_ctrl.cols, axis=-1)
            log_prob_sum += agent_ctrl.log_prob(agent_value)

        return log_prob_sum
        # return tf.reduce_sum(ctrl.log_prob(
        #     tf.gather(value, ctrl.cols, axis=-1)) for ctrl in self.ctrls)

    def update_ctrl(self, errors, u_prev):
        return tf.concat(
            [ctrl.update_ctrl(tf.gather(errors, ctrl.cols, axis=-1),
                              tf.gather(u_prev, ctrl.cols, axis=-1))
             for ctrl in self.ctrls], 0, name='concat_u')

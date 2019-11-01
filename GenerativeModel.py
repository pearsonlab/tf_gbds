import tensorflow as tf
import numpy as np
# import edward as ed
from edward.models import (RandomVariable, MultivariateNormalDiag,
                           ExpRelaxedOneHotCategorical)
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED)


class HHMM(RandomVariable, Distribution):
    def __init__(self, params, states, ctrl_obs, extra_conds, *args,
                 **kwargs):
        name = kwargs.get("name", "HHMM")
        with tf.name_scope(name):
            self.col = params["agent_col"]
            self.dim = params["agent_dim"]
            self.s_dim = params["state_dim"]
            self.extra_dim = params["extra_dim"]
            with tf.name_scope("trial_length"):
                self.Tt = tf.shape(states)[1]
            # self.epoch = epoch
            self.s = tf.identity(states, "states")
            self.y = tf.gather(states, self.col, axis=-1, name="positions")
            self.ctrl_obs = tf.gather(
                ctrl_obs, self.col, axis=-1, name="observed_control")
            self.extra_conds = tf.identity(extra_conds, "extra_conditions")
            self.temperature = params["t_p"]

            self.K_1 = params["K_1"]
            self.K_2 = params["K_2"]
            # neural network to generate state-dependent goals and transition
            self.NN = params["p_NN"]
            NN_outputs = self.NN([self.s, self.extra_conds])

            self.G_mu = tf.identity(NN_outputs[0], "mu")
            self.G_lambda = tf.identity(NN_outputs[1], "lambda")
            self.G = MultivariateNormalDiag(
                self.G_mu[:, -1], 1. / tf.sqrt(self.G_lambda[:, :-1]),
                name="GMM")

            self.exit_probs = tf.identity(NN_outputs[2], "exit_probability")
            self.exit = ExpRelaxedOneHotCategorical(
                temperature=self.temperature, probs=self.exit_probs[:, :-1],
                name="exit_distribution")

            self.z_1_init_logits = params["z_1_init_logits"]
            self.z_1_init_probs = tf.nn.softmax(
                self.z_1_init_logits, name="z_1_initial_probability")
            self.z_1_init = ExpRelaxedOneHotCategorical(
                temperature=self.temperature, logits=self.z_1_init_logits,
                name="z_1_initial_distribution")

            self.z_2_probs = tf.identity(NN_outputs[3], "z_2_probability")
            self.z_2 = ExpRelaxedOneHotCategorical(
                temperature=self.temperature, probs=self.z_2_probs[:, :-1],
                name="z_2_distribution")

            self.unc_Kp = params["unc_Kp"]
            self.Kp = tf.identity(params["Kp"], "Kp")

            self.var_list = (self.NN.variables + [self.z_1_init_logits] +
                             [self.unc_Kp])
            self.log_vars = self.NN.variables

            with tf.name_scope("goal_state_penalty"):
                # penalty on goal state escaping boundaries
                if params["g_bounds_pen"] is not None:
                    self.g_pen = tf.constant(
                        params["g_bounds_pen"], tf.float32,
                        name="goal_boundary_penalty")
                    with tf.name_scope("boundary"):
                        # boundaries for penalty
                        if params["g_bounds"] is not None:
                            self.bounds = params["g_bounds"]
                        else:
                            self.bounds = [-1., 1.]
                else:
                    self.g_pen = None

                # penalty on the precision of GMM components
                # if params["g_prec_pen"] is not None:
                #     self.g_prec_pen = tf.constant(
                #         params["g_prec_pen"], tf.float32,
                #         name="goal_precision_penalty")

            with tf.name_scope("kinetic_noise"):
                # noise of movement
                self.unc_eps = params["unc_eps"]
                self.eps = tf.nn.softplus(self.unc_eps, "epsilon")
                self.eps_pen = params["eps_pen"]
                if params["eps_trainable"]:
                    self.var_list += [self.unc_eps]

        if "name" not in kwargs:
            kwargs["name"] = name
        if "dtype" not in kwargs:
            kwargs["dtype"] = tf.float32
        if "reparameterization_type" not in kwargs:
            kwargs["reparameterization_type"] = FULLY_REPARAMETERIZED
        if "validate_args" not in kwargs:
            kwargs["validate_args"] = True
        if "allow_nan_stats" not in kwargs:
            kwargs["allow_nan_stats"] = False

        super(HHMM, self).__init__(*args, **kwargs)
        self._args = (params, states, ctrl_obs, extra_conds)

    def _log_prob(self, value):
        with tf.name_scope("posterior"):
            qg = tf.identity(value[..., :self.dim], "goals")
            qz_1 = tf.identity(
                value[..., self.dim:(self.dim + self.K_1)], "latent_states_1")
            qz_2 = tf.identity(
                value[..., (self.dim + self.K_1):(
                    self.dim + self.K_1 + self.K_2)], "latent_states_2")
            q_exit = tf.identity(value[..., -2:], "exit_state_probabilities")

        with tf.name_scope("goals"):
            logdensity_g = tf.reduce_mean(tf.reduce_logsumexp(
                tf.add(qz_2, self.G.log_prob(tf.tile(
                    tf.expand_dims(qg, 2), [1, 1, self.K_2, 1]))), -1))

            with tf.name_scope("penalty"):
                if self.g_pen is not None:
                    # penalty on goal state escaping game space
                    logdensity_g -= tf.multiply(
                        self.g_pen, tf.reduce_mean(tf.reduce_sum(
                            tf.nn.relu(self.bounds[0] - self.G_mu) ** 2, -1)))
                    logdensity_g -= tf.multiply(
                        self.g_pen, tf.reduce_mean(tf.reduce_sum(
                            tf.nn.relu(self.G_mu - self.bounds[1]) ** 2, -1)))
                    logdensity_g -= tf.multiply(
                        self.g_pen, tf.reduce_mean(tf.reduce_sum(
                            tf.nn.relu(self.bounds[0] - qg) ** 2, -1)))
                    logdensity_g -= tf.multiply(
                        self.g_pen, tf.reduce_mean(tf.reduce_sum(
                            tf.nn.relu(qg - self.bounds[1]) ** 2, -1)))

                # if self.g_prec_pen is not None:
                #     # penalty on GMM precision
                #     logdensity_g -= tf.multiply(
                #         self.g_prec_pen,
                #         tf.reduce_mean(tf.reduce_max(tf.reduce_sum(
                #             1. / self.G_lambda, -1), -1)))

        with tf.name_scope("latent_states"):
            z_1_cont = MultivariateNormalDiag(
                qz_1[:, :-1], tf.ones([self.K_1]), name="z_1_cont")

            logdensity_z = tf.add_n([
                tf.reduce_mean(tf.reduce_logsumexp(tf.add(
                    qz_1[:, :-1], tf.reduce_logsumexp(tf.add(
                        tf.expand_dims(qz_2[:, :-1], 2),
                        self.exit.log_prob(tf.tile(
                            tf.expand_dims(tf.expand_dims(q_exit, 2), 3),
                            [1, 1, self.K_1, self.K_2, 1]))[:, 1:]), -1)), -1)),
                tf.reduce_mean(tf.reduce_logsumexp(
                    tf.add(q_exit[:, 1:], tf.stack([
                        self.z_1_init.log_prob(qz_1[:, 1:]),
                        z_1_cont.log_prob(qz_1[:, 1:])
                        ], -1)), -1)),
                tf.reduce_mean(tf.reduce_logsumexp(tf.add(
                    qz_1[:, 1:], tf.reduce_logsumexp(tf.add(
                        tf.expand_dims(qz_2[:, :-1], 2),
                        self.z_2.log_prob(tf.tile(
                            tf.expand_dims(tf.expand_dims(qz_2, 2), 3),
                            [1, 1, self.K_1, self.K_2, 1]))[:, 1:]), -1)), -1))
                ])

        with tf.name_scope("control_signal"):
            u_pred = tf.multiply(
                qg - self.y[:, :-1], self.Kp, "predicted")
            u_res = tf.subtract(
                u_pred, self.ctrl_obs[:, 1:], "residual")
            logdensity_u = -tf.reduce_mean(tf.reduce_sum(
                (0.5 * tf.log(2 * np.pi) + tf.log(self.eps) +
                 u_res ** 2 / (2 * self.eps ** 2)), -1))

            with tf.name_scope("penalty"):
                if self.eps_pen is not None:
                    logdensity_u -= self.eps_pen * tf.reduce_sum(self.unc_eps)

        return tf.add_n([logdensity_g, logdensity_z, logdensity_u])

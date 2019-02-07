"""
Base class for a generative model and linear dynamical system implementation.
Based on Evan Archer"s code here:
https://github.com/earcher/vilds/blob/master/code/RecognitionModel.py
"""

import tensorflow as tf
import numpy as np
import lib.blk_tridiag_chol_tools as blk
from edward.models import RandomVariable, Dirichlet
from tensorflow.contrib.distributions import Distribution
from tensorflow.contrib.distributions import FULLY_REPARAMETERIZED


# def get_alpha_rec(NN, Input, extra_conds, yDim, extra_dim, lag,
#                   name="q_alpha"):
#     with tf.name_scope("pad_lag"):
#         Input_ = tf.identity(Input)
#         for i in range(lag):
#             lagged = tf.concat(
#                 [tf.reshape(Input_[:, 0, :yDim], [-1, 1, yDim], "t0"),
#                  Input_[:, :-1, -yDim:]], 1, "lagged")
#             Input_ = tf.concat([Input_, lagged], -1)

#     with tf.name_scope(name):
#         no_second_npc = tf.reduce_all(tf.equal(
#             tf.gather(extra_conds, [extra_dim - 1], axis=-1), 0),
#             name="second_npc_bool")
#         input_1 = tf.concat([Input_, extra_conds], -1, "NN_input_1")
#         input_2 = tf.concat([Input_, tf.concat(
#             [extra_conds[:, :, (extra_dim // 2):],
#              extra_conds[:, :, :(extra_dim // 2)]], -1, "swap_npc")],
#             -1, "NN_input_2")
#         O1 = NN(input_1)
#         O2 = NN(input_2)
#         A0 = (tf.gather(O1, [0], axis=-1) + tf.gather(O2, [0], axis=-1)) / 2.
#         A1 = tf.gather(O1, [1], axis=-1)
#         A2 = tf.gather(O2, [1], axis=-1)

#         params = tf.cond(
#             no_second_npc, lambda: tf.nn.softplus(tf.concat([A0, A1], -1)),
#             lambda: tf.nn.softplus(tf.concat([A0, A1, A2], -1)),
#             name="params")
#         alpha = Dirichlet(params)

#     return NN.variables, params, alpha


# class Alpha_rec(Dirichlet):
#     def __init__(self, NN, traj, extra_conds, yDim, extra_dim, lag, *args,
#                  **kwargs):
#         name = kwargs.get("name", "q_alpha")
#         with tf.name_scope(name):
#             with tf.name_scope("pad_lag"):
#                 Input = tf.identity(traj)
#                 for i in range(lag):
#                     lagged = tf.concat(
#                         [tf.reshape(Input[:, 0, :yDim], [-1, 1, yDim], "t0"),
#                          Input[:, :-1, -yDim:]], 1, "lagged")
#                     Input = tf.concat([Input, lagged], -1)

#             self.traj = tf.identity(Input, "lagged_traj")
#             self.extra_conds = tf.identity(extra_conds, "extra_conds")
#             self.NN = NN
#             self.vars = NN.variables
#             self.d = yDim
#             self.extra_dim = extra_dim

#             no_second_npc = tf.reduce_all(tf.equal(
#                 tf.gather(self.extra_conds, [self.extra_dim - 1], axis=-1),
#                 0), name="second_npc_bool")
#             input_1 = tf.concat(
#                 [self.traj, self.extra_conds], -1, "NN_input_1")
#             input_2 = tf.concat([self.traj, tf.concat(
#                 [self.extra_conds[:, :, (self.extra_dim // 2):],
#                  self.extra_conds[:, :, :(self.extra_dim // 2)]],
#                 -1, "swap_npc")], -1, "NN_input_2")
#             O1 = self.NN(input_1)
#             O2 = self.NN(input_2)
#             A0 = (tf.gather(O1, [0], axis=-1) +
#                   tf.gather(O2, [0], axis=-1)) / 2.
#             A1 = tf.gather(O1, [1], axis=-1)
#             A2 = tf.gather(O2, [1], axis=-1)

#             concentration = tf.cond(
#                 no_second_npc,
#                 lambda: tf.nn.softplus(tf.concat([A0, A1], -1)),
#                 lambda: tf.nn.softplus(tf.concat([A0, A1, A2], -1)),
#                 name="concentration")

#             if "name" not in kwargs:
#                 kwargs["name"] = name
#             # if "dtype" not in kwargs:
#             #     kwargs["dtype"] = tf.float32
#             # if "reparameterization_type" not in kwargs:
#             #     kwargs["reparameterization_type"] = FULLY_REPARAMETERIZED
#             if "validate_args" not in kwargs:
#                 kwargs["validate_args"] = True
#             if "allow_nan_stats" not in kwargs:
#                 kwargs["allow_nan_stats"] = False

#             super(Alpha_rec, self).__init__(concentration, *args, **kwargs)
#             self._args = (NN, traj, extra_conds, yDim, extra_dim, lag)


class SmoothingLDSTimeSeries(RandomVariable, Distribution):
    """A "smoothing" recognition model to approximate the posterior and create
    the appropriate sampling expression given some observations.
    Neural networks are used to parameterize mean and precision (approximated
    by tridiagonal matrix/tensor).

    x ~ N( mu(y), sigma(y) )

    """
    def __init__(self, params, Input, xDim, yDim, extra_conds, *args,
                 **kwargs):
    # def __init__(self, params, Input, xDim, yDim, extra_conds, trainable,
    #              *args, **kwargs):
        """Initialize SmoothingLDSTimeSeries random variable (batch)

        Args:
            params: A Dictionary.
                Dictionary of time series-specific parameters. Contents:
                    * A: linear dynamics matrix (eigeninitial_values with
                         magnitude strictly less than 1);
                    * QinvChol: square root of the innovation covariance
                                matrix Q inverse;
                    * Q0invChol: square root of the initial innovation
                                 covariance matrix Q0 inverse;
                    * Neural network parameters: NN_Mu, NN_Lambda, NN_LambdaX.
            Input: A Tensor. Observations based on which samples are drawn.
            xDim, yDim: Integers. Dimension of latent space (x) and
                        observation (y).
            name: Optional name for the random variable.
                  Default to "SmoothingLDSTimeSeries".
        """
        name = kwargs.get("name", "SmoothingLDSTimeSeries")
        with tf.name_scope(name):
            self.y_t = tf.identity(Input, "observations")
            self.dyn_params = params["dyn_params"]
            self.xDim = xDim
            self.yDim = yDim
            self.extra_dim = params["extra_dim"]
            with tf.name_scope("batch_size"):
                self.B = tf.shape(Input)[0]
            with tf.name_scope("trial_length"):
                self.Tt = tf.shape(Input)[1]

            with tf.name_scope("pad_extra_conds"):
                if extra_conds is not None:
                    self.extra_conds = tf.identity(
                        extra_conds, "extra_conditions")
                    self.y = tf.concat([self.y_t, self.extra_conds], -1)
                else:
                    self.extra_conds = None

            # self.trainable = trainable

            self.NN_Mu = params["NN_Mu"]
            # Mu will automatically be of size [Batch_size x T x xDim]
            self.Mu = tf.identity(self.NN_Mu(self.y), "Mu")
            # self.Mu = tf.identity(tf.cond(
            #     self.trainable, lambda: self.NN_Mu(self.y),
            #     lambda: tf.stop_gradient(self.NN_Mu(self.y))), "Mu")

            self.NN_Lambda = params["NN_Lambda"]
            self.NN_Lambda_output = tf.identity(
                self.NN_Lambda(self.y), "flattened_LambdaChol")
            # self.NN_Lambda_output = tf.identity(tf.cond(
            #     self.trainable, lambda: self.NN_Lambda(self.y),
            #     lambda: tf.stop_gradient(self.NN_Lambda(self.y))),
            #     "flattened_LambdaChol")
            self.LambdaChol = tf.reshape(
                self.NN_Lambda_output, [self.B, self.Tt, xDim, xDim],
                "LambdaChol")

            self.NN_LambdaX = params["NN_LambdaX"]
            self.NN_LambdaX_output = tf.identity(
                self.NN_LambdaX(self.y[:, 1:]), "flattened_LambdaXChol")
            # self.NN_LambdaX_output = tf.identity(tf.cond(
            #     self.trainable, lambda: self.NN_LambdaX(self.y[:, 1:]),
            #     lambda: tf.stop_gradient(self.NN_LambdaX(self.y[:, 1:]))),
            #     "flattened_LambdaXChol")
            self.LambdaXChol = tf.reshape(
                self.NN_LambdaX_output, [self.B, self.Tt - 1, xDim, xDim],
                "LambdaXChol")

            with tf.name_scope("init_posterior"):
                self._initialize_posterior_distribution(params)

            self.NN_Alpha = params["NN_Alpha"]
            self.alpha = self.get_alpha()

            var_list = (self.NN_Mu.variables + self.NN_Lambda.variables +
                        self.NN_LambdaX.variables + self.NN_Alpha.variables +
                        [self.A] + [self.QinvChol] + [self.Q0invChol])
            self.var_list = var_list
            self.log_vars = var_list

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

        super(SmoothingLDSTimeSeries, self).__init__(*args, **kwargs)

        self._args = (params, Input, xDim, yDim, extra_conds)
        # self._args = (params, Input, xDim, yDim, extra_conds, trainable)

    def _initialize_posterior_distribution(self, params):
        # Compute the precisions (from square roots)
        self.Lambda = tf.matmul(self.LambdaChol, tf.transpose(
            self.LambdaChol, [0, 1, 3, 2]), name="Lambda")
        self.LambdaX = tf.matmul(self.LambdaXChol, tf.transpose(
            self.LambdaXChol, [0, 1, 3, 2]), name="LambdaX")

        # Create dynamics matrix & initialize innovations precision,
        # [Batch_size x xDim x xDim]
        with tf.name_scope("dynamics_matrix"):
            self.A = self.dyn_params["A"]

        with tf.name_scope("initialize_innovations_precision"):
            self.QinvChol = self.dyn_params["QinvChol"]
            self.Qinv = tf.matmul(self.QinvChol, tf.transpose(
                self.QinvChol), name="Qinv")
            # self.Qinv = tf.identity(tf.cond(
            #     self.trainable,
            #     lambda: tf.matmul(
            #         self.QinvChol, tf.transpose(self.QinvChol)),
            #     lambda: tf.stop_gradient(tf.matmul(
            #         self.QinvChol, tf.transpose(self.QinvChol)))), "Qinv")

            self.Q0invChol = self.dyn_params["Q0invChol"]
            self.Q0inv = tf.matmul(self.Q0invChol, tf.transpose(
                self.Q0invChol), name="Q0inv")
            # self.Q0inv = tf.identity(tf.cond(
            #     self.trainable,
            #     lambda: tf.matmul(
            #         self.Q0invChol, tf.transpose(self.Q0invChol)),
            #     lambda: tf.stop_gradient(tf.matmul(
            #         self.Q0invChol, tf.transpose(self.Q0invChol)))), "Q0inv")

        with tf.name_scope("noise_penalty"):
            if "p" in params:
                self.p = tf.constant(params["p"], tf.float32, name="p")
            else:
                self.p = None

        # Put together the total precision matrix
        with tf.name_scope("precision_matrix"):
            AQinvA = tf.matmul(tf.matmul(tf.transpose(self.A), self.Qinv),
                               self.A, name="AQinvA")
            # AQinvA = tf.identity(tf.cond(
            #     self.trainable,
            #     lambda: tf.matmul(tf.matmul(
            #         tf.transpose(self.A), self.Qinv), self.A),
            #     lambda: tf.stop_gradient(tf.matmul(tf.matmul(
            #         tf.transpose(self.A), self.Qinv), self.A))), "AQinvA")
            AQinvrep = tf.tile(tf.expand_dims(
                -tf.matmul(tf.transpose(self.A), self.Qinv, name="AQinv"), 0),
                [self.Tt - 1, 1, 1], "AQinvrep")
            # AQinv = tf.identity(tf.cond(
            #     self.trainable,
            #     lambda: tf.matmul(tf.transpose(self.A), self.Qinv),
            #     lambda: tf.stop_gradient(
            #         tf.matmul(tf.transpose(self.A), self.Qinv))), "AQinv")
            # AQinvrep = tf.tile(tf.expand_dims(-AQinv, 0),
            #                    [self.Tt - 1, 1, 1], "AQinvrep")
            AQinvArep = tf.tile(tf.expand_dims(AQinvA + self.Qinv, 0),
                                [self.Tt - 2, 1, 1], "AQinvArep")
            AQinvArepPlusQ = tf.concat(
                [tf.expand_dims(self.Q0inv + AQinvA, 0), AQinvArep,
                 tf.expand_dims(self.Qinv, 0)], 0, "AQinvArepPlusQ")

            with tf.name_scope("diagonal_blocks"):
                self.AA = tf.add(self.Lambda + tf.pad(
                    self.LambdaX, [[0, 0], [1, 0], [0, 0], [0, 0]]),
                    AQinvArepPlusQ, "diagonal")
            with tf.name_scope("off-diagonal_blocks"):
                self.BB = tf.add(tf.matmul(
                    self.LambdaChol[:, :-1],
                    tf.transpose(self.LambdaXChol, [0, 1, 3, 2])),
                    AQinvrep, "off_diagonal")

        with tf.name_scope("posterior_mean"):
            # scale by precision
            LambdaMu = tf.matmul(self.Lambda, tf.expand_dims(self.Mu, -1),
                                 name="Lambda_Mu")

            # compute cholesky decomposition
            self.the_chol = blk.blk_tridiag_chol(self.AA, self.BB)
            # intermediary (mult by R^T)
            ib = blk.blk_chol_inv(self.the_chol[0], self.the_chol[1],
                                  LambdaMu)
            # final result (mult by R)
            self.postX = blk.blk_chol_inv(self.the_chol[0], self.the_chol[1],
                                          ib, lower=False, transpose=True)

        """The determinant of covariance matrix is the square of the
        determinant of Cholesky factor, which is the product of the diagonal
        elements of the block-diagonal.
        """
        with tf.name_scope("log_determinant"):
            def comp_log_det(acc, inputs):
                L = inputs[0]
                return tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), -1)

            self.ln_determinant = -2 * tf.reduce_sum(tf.transpose(tf.scan(
                comp_log_det,
                [tf.transpose(self.the_chol[0], [1, 0, 2, 3])],
                initializer=tf.zeros([self.B]))), -1, name="log_determinant")

    def get_alpha(self, name="q_alpha"):
        with tf.name_scope(name):
            no_second_npc = tf.reduce_all(tf.equal(
                tf.gather(self.extra_conds, [self.extra_dim - 1], axis=-1),
                0), name="second_npc_bool")
            input_1 = tf.concat(
                [self.y_t, self.extra_conds], -1, "NN_input_1")
            input_2 = tf.concat([self.y_t, tf.concat(
                [self.extra_conds[:, :, (self.extra_dim // 2):],
                 self.extra_conds[:, :, :(self.extra_dim // 2)]],
                -1, "swap_npc")], -1, "NN_input_2")
            O1 = self.NN_Alpha(input_1)
            O2 = self.NN_Alpha(input_2)
            A0 = (tf.gather(O1, [0], axis=-1) +
                  tf.gather(O2, [0], axis=-1)) / 2.
            A1 = tf.gather(O1, [1], axis=-1)
            A2 = tf.gather(O2, [1], axis=-1)

            concentration = tf.cond(
                no_second_npc,
                lambda: tf.nn.softplus(tf.concat([A0, A1], -1)),
                lambda: tf.nn.softplus(tf.concat([A0, A1, A2], -1)),
                name="concentration")
            alpha = Dirichlet(concentration)

        return alpha

    def _sample_n(self, n, seed=None):
        return tf.squeeze(tf.map_fn(self.get_sample, tf.zeros([n, self.B])),
                          -1, "samples")

    def get_sample(self, _=None):
        norm_samp = tf.random_normal([self.B, self.Tt, self.xDim],
                                     name="standard_normal_samples")
        return tf.add(blk.blk_chol_inv(
            self.the_chol[0], self.the_chol[1], tf.expand_dims(norm_samp, -1),
            lower=False, transpose=True), self.postX)

    def _log_prob(self, value):
        return tf.reduce_mean(self.eval_entropy())

    def eval_entropy(self):
        # Compute the entropy of LDS (analogous to prior on smoothness)
        entropy = (self.ln_determinant / 2. +
                   tf.cast(self.xDim * self.Tt, tf.float32) / 2. *
                   (1 + np.log(2 * np.pi)))

        # penalize noise
        if self.p is not None:
            entropy += (self.p *
                        (tf.reduce_sum(tf.log(tf.diag_part(self.Qinv))) +
                            tf.reduce_sum(tf.log(tf.diag_part(self.Q0inv)))))

        return entropy / tf.cast(self.Tt, tf.float32)


class SmoothingPastLDSTimeSeries(SmoothingLDSTimeSeries):
    """SmoothingLDSTimeSeries that uses past observations (lag) in addition to
    current to evaluate the latent.
    """
    def __init__(self, params, Input, xDim, yDim, extra_conds, *args,
                 **kwargs):
    # def __init__(self, params, Input, xDim, yDim, extra_conds, trainable,
    #              *args, **kwargs):
        """Initialize SmoothingPastLDSTimeSeries random variable (batch)
        """
        name = kwargs.get("name", "SmoothingPastLDSTimeSeries")
        with tf.name_scope(name):
            with tf.name_scope("pad_lag"):
                # manipulate input to include past observations (up to lag)
                if "lag" in params:
                    self.lag = params["lag"]
                else:
                    self.lag = 1

                Input_ = tf.identity(Input)
                for i in range(self.lag):
                    lagged = tf.concat(
                        [tf.reshape(Input_[:, 0, :yDim], [-1, 1, yDim], "t0"),
                         Input_[:, :-1, -yDim:]], 1, "lagged")
                    Input_ = tf.concat([Input_, lagged], -1)

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

        super(SmoothingPastLDSTimeSeries, self).__init__(
            params, Input_, xDim, yDim, extra_conds, *args, **kwargs)
        # super(SmoothingPastLDSTimeSeries, self).__init__(
        #     params, Input_, xDim, yDim, extra_conds, trainable,
        #     *args, **kwargs)

        self._args = (params, Input, xDim, yDim, extra_conds)
        # self._args = (params, Input, xDim, yDim, extra_conds, trainable)


# class joint_recognition(RandomVariable, Distribution):
#     def __init__(self, q_g_params, q_u_params, trajectory, model_dim, obs_dim,
#                  extra_conds, q_g_trainable, q_u_trainable, *args, **kwargs):
#         name = kwargs.get("name", "joint_recognition")
#         with tf.name_scope(name):
#             self.g = SmoothingPastLDSTimeSeries(
#                 q_g_params, trajectory, model_dim, obs_dim, extra_conds,
#                 q_g_trainable, name="q_g")
#             self.u = SmoothingPastLDSTimeSeries(
#                 q_u_params, trajectory, model_dim, obs_dim, extra_conds,
#                 q_u_trainable, name="q_u")
#             self.names = [self.g.name, self.u.name]
#             self.var_list = self.g.var_list + self.u.var_list
#             self.log_vars = self.g.log_vars + self.u.log_vars

#         if "name" not in kwargs:
#             kwargs["name"] = name
#         if "dtype" not in kwargs:
#             kwargs["dtype"] = tf.float32
#         if "reparameterization_type" not in kwargs:
#             kwargs["reparameterization_type"] = FULLY_REPARAMETERIZED
#         if "validate_args" not in kwargs:
#             kwargs["validate_args"] = True
#         if "allow_nan_stats" not in kwargs:
#             kwargs["allow_nan_stats"] = False

#         super(joint_recognition, self).__init__(*args, **kwargs)
#         self._args = (q_g_params, q_u_params, trajectory, model_dim, obs_dim,
#                       extra_conds, q_g_trainable, q_u_trainable)

#     def _sample_n(self, n, seed=None):
#         return tf.concat([self.g.sample(n), self.u.sample(n)], -1, "samples")

#     def _log_prob(self, value):
#         return tf.add(
#             self.g.log_prob(tf.gather(value, tf.range(self.g.xDim), axis=-1)),
#             self.u.log_prob(tf.gather(
#                 value, self.g.xDim + tf.range(self.u.xDim), axis=-1)))

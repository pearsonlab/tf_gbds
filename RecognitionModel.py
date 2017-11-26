"""
Base class for a generative model and linear dynamical system implementation.
Based on Evan Archer's code here:
https://github.com/earcher/vilds/blob/master/code/RecognitionModel.py
"""
import tensorflow as tf
import numpy as np
import lib.blk_tridiag_chol_tools as blk
from edward.models import RandomVariable
from tensorflow.contrib.distributions import Distribution
from tensorflow.contrib.distributions import FULLY_REPARAMETERIZED
from utils import pad_extra_conds


class SmoothingLDSTimeSeries(RandomVariable, Distribution):
    """A "smoothing" Recognition Model

    Recognition model approximates the posterior given some observations

    Different forms of recognition models will have this interface

    The constructor accepts neural networks which are used to parameterize mu
    and Sigma, and create the appropriate sampling expression.

    x ~ N( mu(y), sigma(y) )

    """

    def __init__(self, RecognitionParams, Input, extra_conds, xDim, yDim, Dyn_params,
                 nrng=None, name="SmoothingLDSTimeSeries", value=None,
                 dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):
        """Initialize a batch of SmoothingLDSTimeSeries random variables


        Args:
          RecognitionParams: (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * A: [Batch_size x n x n] linear dynamics matrix; should
                          have eigeninitial_values with magnitude strictly
                          less than 1
                     * NN paramers: NN_Mu, NN_Lambda, NN_LambdaX
                     * QinvChol: [Batch_size x n x n] square root of the
                                 innovation covariance Q inverse
                     * Q0invChol: [Batch_size x n x n] square root of the
                                  innitial innovation covariance inverse
          Input: 'y' tensorflow placeholder (n_input)
                Observation matrix based on which we produce q(x)
          Dyn_params: (dictionary)
                Dictionary of dynamical parameters. Contents:
                     * A: [Batch_size x n x n] linear dynamics matrix; should
                          have eigeninitial_values with magnitude strictly
                          less than 1
                     * QinvChol: [Batch_size x n x n] square root of the
                                 innovation covariance Q inverse
                     * Q0invChol: [Batch_size x n x n] square root of the
                                  innitial innovation covariance inverse
          xDim, yDim: (integers) dimension of
                latent space (x) and observation (y)
          value: The Recognition Random Variable sample of goal or control
                 signal
        """
        self.Dyn_params = Dyn_params
        self.nrng = nrng
        self.Input = Input
        self.extra_conds = extra_conds

        with tf.name_scope('dimension'):
            self.xDim = xDim
            self.yDim = yDim
            self.B = tf.shape(Input)[0]

        with tf.name_scope('length'):
            self.Tt = tf.shape(Input)[1]

        with tf.name_scope('pad_extra_conds'):
            if self.extra_conds is not None:
                self.Input = pad_extra_conds(self.Input, self.extra_conds)

        with tf.name_scope('Mu'):
            # This is the neural network that parameterizes the state mean, mu
            self.NN_Mu = RecognitionParams['NN_Mu']['network']
            # Mu will automatically be of size [Batch_size x T x xDim]
            self.Mu = self.NN_Mu(self.Input)

        with tf.name_scope('LambdaChol'):
            self.NN_Lambda = RecognitionParams['NN_Lambda']['network']
            self.lambda_net_out = self.NN_Lambda(self.Input)
        # Lambda will automatically be of size [Batch_size x T x xDim x xDim]
            self.LambdaChol = tf.reshape(self.lambda_net_out,
                                         [self.B, self.Tt, xDim, xDim])

        with tf.name_scope('LambdaXChol'):
            self.NN_LambdaX = RecognitionParams['NN_LambdaX']['network']
            self.lambdaX_net_out = self.NN_LambdaX(self.Input[:, 1:])
            self.LambdaXChol = tf.reshape(self.lambdaX_net_out,
                                          [self.B, self.Tt - 1, xDim, xDim])
        with tf.name_scope('init_posterior'):
            self._initialize_posterior_distribution(RecognitionParams)

        super(SmoothingLDSTimeSeries, self).__init__(
            value=value, name=name, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)
        self._kwargs['RecognitionParams'] = RecognitionParams
        self._kwargs['Input'] = Input
        self._kwargs['extra_conds'] = extra_conds
        self._kwargs['xDim'] = xDim
        self._kwargs['yDim'] = yDim
        self._kwargs['nrng'] = nrng
        self._kwargs['Dyn_params'] = Dyn_params

    def _initialize_posterior_distribution(self, RecognitionParams):

        # Now actually compute the precisions (from their square roots)

        with tf.name_scope('Lambda'):
            self.Lambda = tf.matmul(self.LambdaChol,
                                    tf.transpose(self.LambdaChol,
                                                 perm=[0, 1, 3, 2]))
        with tf.name_scope('LambdaX'):
            self.LambdaX = tf.matmul(self.LambdaXChol,
                                     tf.transpose(self.LambdaXChol,
                                                  perm=[0, 1, 3, 2]))

        # dynamics matrix & initialize the innovations precision,
        # Batch_size x xDim x xDim
        with tf.name_scope('dynamics_matrix'):
            self.A = self.Dyn_params['A']

        with tf.name_scope('init_innovations_prec'):
            with tf.name_scope('Qinv'):
                self.QinvChol = self.Dyn_params['QinvChol']
                self.Qinv = tf.matmul(self.QinvChol,
                                      tf.transpose(self.QinvChol))

            with tf.name_scope('Q0inv'):
                self.Q0invChol = self.Dyn_params['Q0invChol']
                self.Q0inv = tf.matmul(self.Q0invChol,
                                       tf.transpose(self.Q0invChol))

        with tf.name_scope('noise_penalty'):
            if 'p' in RecognitionParams:
                self.p = tf.constant(RecognitionParams['p'],
                                     name='p', dtype=tf.float32)
            else:
                self.p = None

        # put together the total precision matrix #
        with tf.name_scope('prec_matrix'):
            AQinvA = tf.matmul(tf.matmul(tf.transpose(self.A), self.Qinv),
                               self.A, name='AQinvA')
            # for now we (suboptimally) replicate a bunch of times
            AQinvrep = tf.tile(-tf.matmul(tf.transpose(self.A), self.Qinv),
                               [self.Tt - 1, 1])
            # off-diagonal blocks (upper triangle)
            AQinvrep = tf.reshape(AQinvrep,
                                  [self.Tt - 1, self.xDim, self.xDim],
                                  name='AQinvrep')
            self.AQinvrep = AQinvrep
            AQinvArep = tf.tile(AQinvA + self.Qinv, [self.Tt - 2, 1])
            AQinvArep = tf.reshape(AQinvArep,
                                   [self.Tt - 2, self.xDim, self.xDim],
                                   name='AQinvArep')
            AQinvArepPlusQ = tf.concat(
                [tf.expand_dims(self.Q0inv + AQinvA, 0), AQinvArep,
                 tf.expand_dims(self.Qinv, 0)], 0, name='AQinvArepPlusQ')

            # This is our inverse covariance matrix: diagonal (AA)
            # and off-diagonal (BB) blocks.
            with tf.name_scope('diagonal'):
                self.AA = (self.Lambda +
                           tf.concat([tf.zeros([self.B, 1, self.xDim,
                                                self.xDim]), self.LambdaX],
                                     1) + AQinvArepPlusQ)
            with tf.name_scope('off-diagonal'):
                self.BB = (tf.matmul(self.LambdaChol[:, :-1],
                                     tf.transpose(self.LambdaXChol,
                                                  perm=[0, 1, 3, 2])) +
                           AQinvrep)

        with tf.name_scope('posterior_mean'):
            # now compute the posterior mean
            LambdaMu = tf.matmul(self.Lambda, tf.expand_dims(self.Mu, -1))
            # scale by precision (no need for transpose; lambda is symmetric)

            # self.old_postX = sym.compute_sym_blk_tridiag_inv_b(self.S,
            # self.V, LambdaMu) # apply inverse

            with tf.name_scope('chol_decomp'):
                # compute cholesky decomposition
                self.the_chol = blk.blk_tridiag_chol(self.AA, self.BB)
                # intermediary (mult by R^T) -
                ib = blk.blk_chol_inv(self.the_chol[0], self.the_chol[1],
                                      LambdaMu)

            # final result (mult by R)-
            self.postX = blk.blk_chol_inv(self.the_chol[0], self.the_chol[1],
                                          ib, lower=False, transpose=True)

        # The determinant of the covariance is the square of the determinant
        # of the cholesky factor.
        # Determinant of the Cholesky factor is the product of the diagonal
        # elements of the block-diagonal.
        with tf.name_scope('log_determinant'):
            def comp_log_det(acc, inputs):
                L = inputs[0]
                return tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=-1)

            self.ln_determinant = -2 * tf.reduce_sum(tf.transpose(
                tf.scan(comp_log_det,
                        [tf.transpose(self.the_chol[0], perm=[1, 0, 2, 3])],
                        initializer=tf.zeros([self.B]))), axis=-1)

    def _sample_n(self, n, seed=None):
        with tf.name_scope('samples'):
            result = tf.map_fn(self.getSample, tf.zeros([n, self.B]))
        return tf.squeeze(result, -1)

    def getSample(self, _=None):
        """Generate the sample of recognition goal or control signal
        """
        normSamps = tf.random_normal([self.B, self.Tt, self.xDim])
        return self.postX + blk.blk_chol_inv(self.the_chol[0],
                                             self.the_chol[1],
                                             tf.expand_dims(normSamps, -1),
                                             lower=False, transpose=True)

    def _log_prob(self, value):
        return self.evalEntropy()

    def evalEntropy(self):
        """Return the Entropy of model

        # we want it to be smooth, this is a prior on being smooth... #
        """
        entropy = (self.ln_determinant / 2. +
                   tf.cast(self.xDim * self.Tt, tf.float32) / 2. *
                   (1 + np.log(2 * np.pi)))

        if self.p is not None:  # penalize noise
            entropy += (self.p *
                        (tf.reduce_sum(tf.log(tf.diag_part(self.Qinv))) +
                            tf.reduce_sum(tf.log(tf.diag_part(self.Q0inv)))))
        return entropy

    def getDynParams(self):
        """Return the dynamical parameters of the model
        """
        return [self.A] + [self.QinvChol] + [self.Q0invChol]

    def getParams(self):
        """Return the learnable parameters of the model
        """
        return (self.getDynParams() + self.NN_Mu.variables +
                self.NN_Lambda.variables + self.NN_LambdaX.variables)


class SmoothingPastLDSTimeSeries(SmoothingLDSTimeSeries):
    """SmoothingLDSTimeSeries that uses past observations in addition to
    current to evaluate the latent.
    """
    def __init__(self, RecognitionParams, Input, extra_conds, xDim, yDim, Dyn_params,
                 nrng=None, name='SmoothingPastLDSTimeSeries',
                 value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):
        """Initialize a batch of SmoothingPastLDSTimeSeries random variables

        Args:
          RecognitionParams: (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * A: [Batch_size x n x n] linear dynamics matrix; should
                          have eigeninitial_values with magnitude strictly
                          less than 1
                     * NN paramers: NN_Mu, NN_Lambda, NN_LambdaX
                     * QinvChol: [Batch_size x n x n] square root of the
                                 innovation covariance Q inverse
                     * Q0invChol: [Batch_size x n x n] square root of the
                                  innitial innovation covariance inverse
                     * lag
          Input: 'y' tensorflow placeholder (n_input)
                Observation matrix based on which we produce q(x)
          Dyn_params: (dictionary)
                Dictionary of dynamical parameters. Contents:
                     * A: [Batch_size x n x n] linear dynamics matrix; should
                          have eigeninitial_values with magnitude strictly
                          less than 1
                     * QinvChol: [Batch_size x n x n] square root of the
                                 innovation covariance Q inverse
                     * Q0invChol: [Batch_size x n x n] square root of the
                                  innitial innovation covariance inverse
          xDim, yDim: (integers) dimension of
                latent space (x) and observation (y)
          value: The Recognition Random Variable sample of goal or control
                 signal

        """
        with tf.name_scope('lag'):
            if 'lag' in RecognitionParams:
                self.lag = RecognitionParams['lag']
            else:
                self.lag = 5
        # manipulate input to include past observations (up to lag)
        # in each row
        with tf.name_scope('pad_lag'):
            Input_ = Input
            for i in range(self.lag):
                lagged = tf.concat(
                    [tf.reshape(Input_[:, 0, :yDim], [-1, 1, yDim]),
                     Input_[:, :-1, -yDim:]], 1, name='lag')
                Input_ = tf.concat([Input_, lagged], -1, name='pad')

        super(SmoothingPastLDSTimeSeries, self).__init__(
            RecognitionParams, Input_, extra_conds, xDim, yDim, Dyn_params, nrng, name,
            value, dtype, reparameterization_type,
            validate_args, allow_nan_stats)

        self.Input = Input
        self._kwargs['Input'] = Input

"""
Base class for a generative model and linear dynamical system implementation.
Based on Evan Archer's code here:
https://github.com/earcher/vilds/blob/master/code/RecognitionModel.py
"""
import tensorflow as tf
import numpy as np
import tf_gbds.lib.sym_blk_tridiag_inv as sym
import tf_gbds.lib.blk_tridiag_chol_tools as blk
from edward.models import MultivariateNormalTriL
from edward.models import RandomVariable
from tensorflow.contrib.distributions import Distribution
from tensorflow.contrib.distributions import FULLY_REPARAMETERIZED

class SmoothingLDSTimeSeries(RandomVariable, Distribution):
    """
    A "smoothing" recognition model. The constructor accepts neural networks
    which are used to parameterize mu and Sigma.

    x ~ N( mu(y), sigma(y) )

    """

    def __init__(self, RecognitionParams, Input, xDim, yDim, Dyn_params,
                 nrng=None, name="SmoothingLDSTimeSeries_g", value=None,
                 dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):
        """
        :parameters:
            - Input : 'y' tensorflow placeholder (n_input)
                Observation matrix based on which we produce q(x)
            - RecognitionParams : (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * A -
                     * NN paramers...
                     * others... TODO
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x) and observation (y)
        """
        self.Dyn_params = Dyn_params
        self.nrng = nrng
        self.Input = Input
        with tf.name_scope('dimension'):
            self.xDim = xDim
            self.yDim = yDim

        with tf.name_scope('length'):
            self.Tt = tf.shape(Input)[0]

        with tf.name_scope('Mu'):
            # This is the neural network that parameterizes the state mean, mu
            self.NN_Mu = RecognitionParams['NN_Mu']['network']
            # Mu will automatically be of size [T x xDim]
            self.Mu = self.NN_Mu(self.Input)

        with tf.name_scope('LambdaChol'):
            self.NN_Lambda = RecognitionParams['NN_Lambda']['network']
            self.lambda_net_out = self.NN_Lambda(self.Input)
            # Lambda will automatically be of size [T x xDim x xDim]
            self.LambdaChol = tf.reshape(self.lambda_net_out, 
                                         [self.Tt, xDim, xDim])  # + T.eye(self.xDim)

        with tf.name_scope('LambdaXChol'):
            self.NN_LambdaX = RecognitionParams['NN_LambdaX']['network']
            self.lambdaX_net_out = self.NN_LambdaX(self.Input[1:])
            self.LambdaXChol = tf.reshape(self.lambdaX_net_out, 
                                          [self.Tt-1, xDim, xDim])
        with tf.name_scope('init_posterior'):
            self._initialize_posterior_distribution(RecognitionParams)

        super(SmoothingLDSTimeSeries, self).__init__(
            value=value, name=name, dtype=dtype,
            reparameterization_type=reparameterization_type, 
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)
        self._kwargs['RecognitionParams'] = RecognitionParams
        self._kwargs['Input'] = Input
        self._kwargs['xDim'] = xDim
        self._kwargs['yDim'] = yDim
        self._kwargs['nrng'] = nrng
        self._kwargs['Dyn_params'] = Dyn_params

    def _initialize_posterior_distribution(self, RecognitionParams):
        # Now actually compute the precisions (from their square roots)
        with tf.name_scope('Lambda'):
            self.Lambda = tf.matmul(self.LambdaChol, tf.transpose(self.LambdaChol,
                                                                  perm=[0, 2, 1]))
        with tf.name_scope('LambdaX'):
            self.LambdaX = tf.matmul(self.LambdaXChol,
                                 tf.transpose(self.LambdaXChol,
                                              perm=[0, 2, 1]))

        # dynamics matrix & initialize the innovations precision, xDim x xDim
        with tf.name_scope('dynamics_matrix'):
            self.A = self.Dyn_params['A']

        with tf.name_scope('init_innovations_prec'):
            with tf.name_scope('Qinv'):
                self.QinvChol = self.Dyn_params['QinvChol']
                self.Qinv = tf.matmul(self.QinvChol, tf.transpose(self.QinvChol))

            with tf.name_scope('Q0inv'):
                self.Q0invChol = self.Dyn_params['Q0invChol']
                self.Q0inv = tf.matmul(self.Q0invChol, tf.transpose(self.Q0invChol))

        with tf.name_scope('noise_penalty'):
            if 'p' in RecognitionParams:
                self.p = tf.Variable(np.cast[np.float32](RecognitionParams['p']),
                                     name='p')
            else:
                self.p = None

        # put together the total precision matrix #
        with tf.name_scope('prec_matrix'):
            AQinvA = tf.matmul(tf.matmul(tf.transpose(self.A), self.Qinv), self.A)
            # for now we (suboptimally) replicate a bunch of times
            AQinvrep = tf.tile(-tf.matmul(tf.transpose(self.A), self.Qinv),
                               [self.Tt-1, 1])
            # off-diagonal blocks (upper triangle)
            AQinvrep = tf.reshape(AQinvrep, [self.Tt-1, self.xDim, self.xDim])
            self.AQinvrep = AQinvrep
            AQinvArep = tf.tile(AQinvA + self.Qinv, [self.Tt-2, 1])
            AQinvArep = tf.reshape(AQinvArep, [self.Tt-2, self.xDim, self.xDim])
            AQinvArepPlusQ = tf.concat([tf.expand_dims(self.Q0inv + AQinvA, 0),
                                       AQinvArep, tf.expand_dims(self.Qinv, 0)], 0)

            # This is our inverse covariance matrix: diagonal (AA)
            # and off-diagonal (BB) blocks.
            with tf.name_scope('diagonal'):
                self.AA = (self.Lambda + tf.concat([tf.expand_dims(tf.zeros([self.xDim,
                           self.xDim]), 0), self.LambdaX], 0) + AQinvArepPlusQ)
            with tf.name_scope('off-diagonal'):    
                self.BB = (tf.matmul(self.LambdaChol[:-1],
                                     tf.transpose(self.LambdaXChol, perm=[0, 2, 1])) +
                           AQinvrep)
            with tf.name_scope('cov_matrix'):
                # symbolic recipe for computing the the diagonal (V) and
                # off-diagonal (VV) blocks of the posterior covariance
                self.V, self.VV, self.S = sym.compute_sym_blk_tridiag(self.AA, self.BB)

        with tf.name_scope('posterior_mean'):
            # now compute the posterior mean
            LambdaMu = tf.matmul(self.Lambda, tf.reshape(self.Mu,
                                                         [tf.shape(self.Mu)[0],
                                                          tf.shape(self.Mu)[1],
                                                          1]))
            # scale by precision (no need for transpose; lambda is symmetric)

            # self.old_postX = sym.compute_sym_blk_tridiag_inv_b(self.S, self.V,
            # LambdaMu) # apply inverse

            with tf.name_scope('chol_decomp'):
                # compute cholesky decomposition
                self.the_chol = blk.blk_tridiag_chol(self.AA, self.BB)
                # intermediary (mult by R^T) -
                ib = blk.blk_chol_inv(self.the_chol[0], self.the_chol[1], LambdaMu)

            # final result (mult by R)-
            self.postX = blk.blk_chol_inv(self.the_chol[0], self.the_chol[1], ib,
                                          lower=False, transpose=True)

        # The determinant of the covariance is the square of the determinant
        # of the cholesky factor.
        # Determinant of the Cholesky factor is the product of the diagonal
        # elements of the block-diagonal.
        with tf.name_scope('log_determinant'):
            def comp_log_det(acc, inputs):
                L = inputs[0]
                return tf.reduce_sum(tf.log(tf.diag_part(L)))
            self.ln_determinant = -2*tf.reduce_sum(tf.scan(comp_log_det,
                                                   [self.the_chol[0]],
                                                   initializer=0.0))

    def _sample_n(self, n, seed=None):
        with tf.name_scope('posterior_samples'):
            result = tf.map_fn(self.getSample, tf.zeros(n))
        return tf.squeeze(result, -1)

    def getSample(self, _=None):
        normSamps = tf.random_normal([self.Tt, self.xDim])
        return self.postX + blk.blk_chol_inv(self.the_chol[0],
                                             self.the_chol[1],
                                             tf.expand_dims(normSamps, -1),
                                             lower=False, transpose=True)

    def _log_prob(self, value):
        return self.evalEntropy()

    def evalEntropy(self):
        # we want it to be smooth, this is a prior on being smooth... #
        entropy = (self.ln_determinant/2 +
                   tf.cast(self.xDim * self.Tt, tf.float32)/2.0 *
                   (1 + np.log(2 * np.pi)))

        if self.p is not None:  # penalize noise
            entropy += self.p * tf.reduce_sum(tf.log(tf.diag_part(self.Qinv)))
            entropy += self.p * tf.reduce_sum(tf.log(tf.diag_part(self.Q0inv)))
        return entropy

    def getDynParams(self):
        return [self.A]+[self.QinvChol]+[self.Q0invChol]

    def getParams(self):
        return (self.getDynParams() + self.NN_Mu.variables +
                self.NN_Lambda.variables + self.NN_LambdaX.variables)

    def get_summary(self, yy):
        out = {}
        out['xsm'] = np.asarray(self.postX.eval({self.Input: yy}),
                                dtype=np.float32)
        out['Vsm'] = np.asarray(self.V.eval({self.Input: yy}),
                                dtype=np.float32)
        out['VVsm'] = np.asarray(self.VV.eval({self.Input: yy}),
                                 dtype=np.float32)
        out['Mu'] = np.asarray(self.Mu.eval({self.Input: yy}),
                               dtype=np.float32)
        return out

class SmoothingPastLDSTimeSeries(SmoothingLDSTimeSeries):
    """
    SmoothingLDSTimeSeries that uses past observations in addition to current
    to evaluate the latent.
    """
    def __init__(self, RecognitionParams, Input, xDim, yDim, Dyn_params, ntrials, nrng=None, 
      name="SmoothingPastLDSTimeSeries_g", value=None , dtype=tf.float32, 
      reparameterization_type=FULLY_REPARAMETERIZED, validate_args=True, 
      allow_nan_stats=True):
        """
        :parameters:
            - Input : 'y' tensorflow placeholder (n_input)
                Observation matrix based on which we produce q(x)
            - RecognitionParams : (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * A -
                     * NN paramers...
                     * others... TODO
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x) and observation (y)
        """
        with tf.name_scope('n_trials'):
            self.ntrials = np.cast[np.float32](ntrials)
        with tf.name_scope('lag'):
            if 'lag' in RecognitionParams:
                self.lag = RecognitionParams['lag']
            else:
                self.lag = 5

        # manipulate input to include past observations (up to lag)
        # in each row
        with tf.name_scope('pad_lag'):
            for i in range(1, self.lag + 1):
                lagged = tf.concat([tf.reshape(Input[0, :yDim], [1, yDim]),
                                    Input[:-1, -yDim:]], 0, name='lagged')
                Input = tf.concat([Input, lagged], 1)
        super(SmoothingPastLDSTimeSeries, self).__init__(RecognitionParams,
                                                         Input, xDim, yDim,Dyn_params,
                                                         nrng, name, value, dtype,
                                                         reparameterization_type,
                                                         validate_args, allow_nan_stats)
        self._kwargs['ntrials'] = self.ntrials

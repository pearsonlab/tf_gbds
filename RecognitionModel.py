"""
Base class for a generative model and linear dynamical system implementation.
Based on Evan Archer's code here:
https://github.com/earcher/vilds/blob/master/code/RecognitionModel.py
"""
import tensorflow as tf
import numpy as np
import tf_gbds.lib.sym_blk_tridiag_inv as sym
import tf_gbds.lib.blk_tridiag_chol_tools as blk


class RecognitionModel(object):
    """
    Recognition Model Interace Class

    Recognition model approximates the posterior given some observations

    Different forms of recognition models will have this interface

    The constructor must take the Input Theano variable and create the
    appropriate sampling expression.
    """

    def __init__(self, Input, xDim, yDim, srng=None, nrng=None):
        self.srng = srng
        self.nrng = nrng

        self.xDim = xDim
        self.yDim = yDim
        self.Input = Input

    def evalEntropy(self):
        """
        Evaluates entropy of posterior approximation

        H(q(x))

        This is NOT normalized by the number of samples
        """
        raise Exception("Please implement me. This is an abstract method.")

    def getParams(self):
        """
        Returns a list of objects that are parameters of the
        recognition model. These will be updated during learning
        """
        return self.params

    def getSample(self):
        """
        Returns a Theano object that are samples from the recognition model
        given the input
        """
        raise Exception("Please implement me. This is an abstract method.")

    def setTrainingMode(self):
        """
        changes the internal state so that `getSample` will possibly return
        noisy samples for better generalization
        """
        raise Exception("Please implement me. This is an abstract method.")

    def setTestMode(self):
        """
        changes the internal state so that `getSample` will supress noise
        (e.g., dropout) for prediction
        """
        raise Exception("Please implement me. This is an abstract method.")


class SmoothingLDSTimeSeries(RecognitionModel):
    """
    A "smoothing" recognition model. The constructor accepts neural networks
    which are used to parameterize mu and Sigma.

    x ~ N( mu(y), sigma(y) )

    """

    def __init__(self, RecognitionParams, Input, xDim, yDim, srng=None,
                 nrng=None):
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
        super(SmoothingLDSTimeSeries, self).__init__(Input, xDim, yDim, srng,
                                                     nrng)

        self.Tt = tf.shape(Input)[0]

        # This is the neural network that parameterizes the state mean, mu
        self.NN_Mu = RecognitionParams['NN_Mu']['network']

        self.NN_Lambda = RecognitionParams['NN_Lambda']['network']

        self.NN_LambdaX = RecognitionParams['NN_LambdaX']['network']

        # Mu will automatically be of size [T x xDim]
        self.Mu = self.NN_Mu(self.Input)
        self.lambda_net_out = self.NN_Lambda(self.Input)
        self.lambdaX_net_out = self.NN_LambdaX(self.Input[1:])
        # Lambda will automatically be of size [T x xDim x xDim]
        self.LambdaChol = tf.reshape(self.lambda_net_out, [self.Tt, xDim,
                                     xDim])  # + T.eye(self.xDim)
        self.LambdaXChol = tf.reshape(self.lambdaX_net_out, [self.Tt-1, xDim,
                                      xDim])

        self._initialize_posterior_distribution(RecognitionParams)

    def _initialize_posterior_distribution(self, RecognitionParams):

        # Now actually compute the precisions (from their square roots)
        self.Lambda = tf.matmul(self.LambdaChol, tf.transpose(self.LambdaChol,
                                                              perm=[0, 2, 1]))
        self.LambdaX = tf.matmul(self.LambdaXChol,
                                 tf.transpose(self.LambdaXChol,
                                              perm=[0, 2, 1]))

        # dynamics matrix & initialize the innovations precision, xDim x xDim
        self.A = tf.Variable(RecognitionParams['A'].astype(np.float32),
                             name='A')
        self.QinvChol = tf.Variable(RecognitionParams['QinvChol']
                                    .astype(np.float32), name='QinvChol')
        self.Q0invChol = tf.Variable(RecognitionParams['Q0invChol']
                                     .astype(np.float32), name='Q0invChol')

        self.Qinv = tf.matmul(self.QinvChol, tf.transpose(self.QinvChol))
        self.Q0inv = tf.matmul(self.Q0invChol, tf.transpose(self.Q0invChol))

        if 'p' in RecognitionParams:
            self.p = tf.Variable(np.cast[np.float32](RecognitionParams['p']),
                                 name='p')
        else:
            self.p = None

        # put together the total precision matrix #

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
        self.AA = (self.Lambda + tf.concat([tf.expand_dims(tf.zeros([self.xDim,
                   self.xDim]), 0), self.LambdaX], 0) + AQinvArepPlusQ)
        self.BB = (tf.matmul(self.LambdaChol[:-1],
                             tf.transpose(self.LambdaXChol, perm=[0, 2, 1])) +
                   AQinvrep)

        # symbolic recipe for computing the the diagonal (V) and
        # off-diagonal (VV) blocks of the posterior covariance
        self.V, self.VV, self.S = sym.compute_sym_blk_tridiag(self.AA, self.BB)

        # now compute the posterior mean
        LambdaMu = tf.matmul(self.Lambda, tf.reshape(self.Mu,
                                                     [tf.shape(self.Mu)[0],
                                                      tf.shape(self.Mu)[1],
                                                      1]))
        # scale by precision (no need for transpose; lambda is symmetric)

        # self.old_postX = sym.compute_sym_blk_tridiag_inv_b(self.S, self.V,
        # LambdaMu) # apply inverse

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
        def comp_log_det(acc, inputs):
            L = inputs[0]
            return tf.reduce_sum(tf.log(tf.diag_part(L)))
        self.ln_determinant = -2*tf.reduce_sum(tf.scan(comp_log_det,
                                               [self.the_chol[0]],
                                               initializer=0.0))
        self.scan = tf.scan(comp_log_det, [self.the_chol[0]], initializer=0.0)

    def getSample(self):
        normSamps = tf.random_normal([self.Tt, self.xDim])
        return self.postX + blk.blk_chol_inv(self.the_chol[0],
                                             self.the_chol[1],
                                             tf.expand_dims(normSamps, -1),
                                             lower=False, transpose=True)

    def evalEntropy(self):
        # we want it to be smooth, this is a prior on being smooth... #
        entropy = (self.ln_determinant/2 + self.xDim *
                   tf.cast(self.Tt, tf.float32)/2.0 * (1 + np.log(2 * np.pi)))

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
    def __init__(self, RecognitionParams, Input, xDim, yDim, ntrials,
                 srng=None, nrng=None):
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
        self.ntrials = np.cast[np.float32](ntrials)
        if 'lag' in RecognitionParams:
            self.lag = RecognitionParams['lag']
        else:
            self.lag = 5

        # manipulate input to include past observations (up to lag) in each row
        for i in range(1, self.lag + 1):
            lagged = tf.concat([tf.reshape(Input[0, :yDim], [1, yDim]),
                               Input[:-1, -yDim:]], 0)
            Input = tf.concat([Input, lagged], 1)
        self.Input1 = Input
        super(SmoothingPastLDSTimeSeries, self).__init__(RecognitionParams,
                                                         Input, xDim, yDim,
                                                         srng, nrng)


class SmoothingTimeSeries(RecognitionModel):
    '''
    A "smoothing" recognition model. The constructor accepts neural networks
    which are used to parameterize mu and Sigma.

    x ~ N( mu(y), sigma(y) )

    '''

    def __init__(self, RecognitionParams, Input, xDim, yDim, srng=None,
                 nrng=None):
        '''
        :parameters:
            - Input : 'y' theano.tensor.var.TensorVariable (n_input)
                Observation matrix based on which we produce q(x)
            - RecognitionParams : (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * A -
                     * NN paramers...
                     * others... TODO
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x) and observation (y)
        '''
        super(SmoothingTimeSeries, self).__init__(Input, xDim, yDim, srng,
                                                  nrng)

#        print RecognitionParams

        self.Tt = tf.shape(Input)[0]
        # These variables allow us to control whether the network is
        # deterministic or not (if we use Dropout)
        self.mu_train = RecognitionParams['NN_Mu']['is_train']
        self.lambda_train = RecognitionParams['NN_Lambda']['is_train']

        # This is the neural network that parameterizes the state mean, mu
        self.NN_Mu = RecognitionParams['NN_Mu']['network']
        # Mu will automatically be of size [T x xDim]
        self.Mu = self.NN_Mu(self.Input)

        self.NN_Lambda = RecognitionParams['NN_Lambda']['network']
        lambda_net_out = self.NN_Lambda(self.Input)

        self.NN_LambdaX = RecognitionParams['NN_LambdaX']['network']
        lambdaX_net_in = tf.concat([self.Input[:-1], self.Input[1:]], axis=1)
        lambdaX_net_out = self.NN_LambdaX(lambdaX_net_in)

        # Lambda will automatically be of size [T x xDim x xDim]
        self.AAChol = (tf.reshape(lambda_net_out, [self.Tt, xDim, xDim]) +
                       tf.eye(xDim))
        self.BBChol = tf.reshape(lambdaX_net_out, [self.Tt-1, xDim, xDim])
        # + 1e-6*tf.eye(xDim)

        self._initialize_posterior_distribution(RecognitionParams)

    def _initialize_posterior_distribution(self, RecognitionParams):

        # put together the total precision matrix

        # Diagonals must be PSD
        diagsquare = tf.matmul(self.AAChol, tf.transpose(self.AAChol,
                                                         perm=[0, 2, 1]))
        odsquare = tf.matmul(self.BBChol, tf.transpose(self.BBChol,
                                                       perm=[0, 2, 1]))
        self.AA = (diagsquare + tf.concat(
            [tf.expand_dims(tf.zeros([self.xDim, self.xDim]), 0),
             odsquare], axis=0) + 1e-6*tf.eye(self.xDim))
        self.BB = tf.matmul(self.AAChol[:-1], tf.transpose(self.BBChol,
                                                           perm=[0, 2, 1]))

        # compute Cholesky decomposition
        self.the_chol = blk.blk_tridiag_chol(self.AA, self.BB)

        # symbolic recipe for computing the the diagonal (V) and
        # off-diagonal (VV) blocks of the posterior covariance
        self.V, self.VV, self.S = sym.compute_sym_blk_tridiag(self.AA,
                                                              self.BB)
        self.postX = self.Mu

        # The determinant of the covariance is the square of the determinant
        # of the cholesky factor (twice the log).
        # Determinant of the Cholesky factor is the product of the diagonal
        # elements of the block-diagonal.
        def comp_log_det(acc, inputs):
            L = inputs[0]
            return tf.reduce_sum(tf.log(tf.diag_part(L)))
        self.ln_determinant = -2*tf.reduce_sum(tf.scan(comp_log_det,
                                               [self.the_chol[0]],
                                               initializer=0.0))

    def getSample(self):
        normSamps = tf.random_normal([self.Tt, self.xDim])
        return self.postX + blk.blk_chol_inv(self.the_chol[0],
                                             self.the_chol[1],
                                             tf.expand_dims(normSamps, -1),
                                             lower=False, transpose=True)

    def evalEntropy(self):
        return (self.ln_determinant/2 +
                self.xDim*self.Tt/2.0*(1+np.log(2*np.pi)))

    def getParams(self):
        return (self.NN_Mu.variables + self.NN_Lambda.variables +
                self.NN_LambdaX.variables)

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


class MeanFieldGaussian(RecognitionModel):
    '''
    Define a mean field variational approximate posterior (Recognition Model).
    Here, "mean field" is over time, so that for
    x = (x_1, \dots, x_t, \dots, x_T):

    x ~ \prod_{t=1}^T N( mu_t(y_t), sigma_t(y_t) ).

    Each covariance sigma_t is a full [n x n] covariance matrix (where n is
    the size of the latent space).

    '''

    def __init__(self, RecognitionParams, Input, xDim, yDim, srng=None,
                 nrng=None):
        '''
        :parameters:
            - Input : 'y' theano.tensor.var.TensorVariable (n_input)
                Observation matrix based on which we produce q(x)
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x), observation (y)
        '''
        super(MeanFieldGaussian, self).__init__(Input, xDim, yDim, srng, nrng)
        self.Tt = tf.shape(Input)[0]
        self.mu_train = RecognitionParams['NN_Mu']['is_train']
        self.NN_Mu = RecognitionParams['NN_Mu']['network']
        self.postX = self.NN_Mu(self.Input)

        self.lambda_train = RecognitionParams['NN_Lambda']['is_train']
        self.NN_Lambda = RecognitionParams['NN_Lambda']['network']

        lambda_net_out = self.NN_Lambda(self.Input)
        self.LambdaChol = tf.reshape(lambda_net_out, [self.Tt, xDim, xDim])

    def getParams(self):
        network_params = self.NN_Mu.variables + self.NN_Lambda.variables
        return network_params

    def evalEntropy(self):
        def compTrace(Rt):
            return tf.log(tf.abs(tf.matrix_determinant(Rt)))
        theDet, updates = tf.scan(fn=compTrace, elems=[self.LambdaChol])
        return (tf.reduce_sum(theDet) +
                self.xDim*self.Tt/2.0 * (1 + np.log(2*np.pi)))

    def getSample(self):

        normSamps = tf.random_normal([self.Tt, self.xDim])

        def SampOneStep(SampRt, nsampt):
            return tf.matmul(nsampt, tf.transpose(SampRt))
        retSamp = tf.scan(fn=SampOneStep,
                          elems=[self.LambdaChol, normSamps])[0]
        return retSamp + self.postX

    def get_summary(self, yy):
        out = {}
        out['xsm'] = np.asarray(self.postX.eval({self.Input: yy}),
                                dtype=np.float32)
        V = tf.matmul(self.LambdaChol, tf.transpose(self.LambdaChol,
                                                    perm=[0, 2, 1]))
        out['Vsm'] = np.asarray(V.eval({self.Input: yy}), dtype=np.float32)
        out['VVsm'] = np.zeros([yy.shape[0]-1, self.xDim, self.xDim],
                               dtype=np.float32)
        return out

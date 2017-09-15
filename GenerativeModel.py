"""
Base class for a generative model and linear dynamical system implementation.
Based on Evan Archer's code here: https://github.com/earcher/vilds/blob/master/code/GenerativeModel.py
"""
import tensorflow as tf
import numpy as np


class GenerativeModel(object):
    """
    Interface class for generative time-series models
    """
    def __init__(self, GenerativeParams, xDim, yDim, srng=None, nrng=None):

        # input variable referencing top-down or external input

        self.xDim = xDim
        self.yDim = yDim

        self.srng = srng
        self.nrng = nrng

        # internal RV for generating sample
        self.Xsamp = tf.placeholder(tf.float32,
                                    shape=(None, xDim), name='Xsamp')

    def evaluateLogDensity(self):
        """
        Return a function that evaluates the density of the GenerativeModel.
        """
        raise Exception('Cannot call function of interface class')

    def getParams(self):
        """
        Return parameters of the GenerativeModel.
        """
        raise Exception('Cannot call function of interface class')

    def generateSamples(self):
        """
        generates joint samples
        """
        raise Exception('Cannot call function of interface class')

    def __repr__(self):
        return "GenerativeModel"


class LDS(GenerativeModel):
    """
    Gaussian latent LDS with (optional) NN observations:

    x(0) ~ N(x0, Q0 * Q0')
    x(t) ~ N(A x(t-1), Q * Q')
    y(t) ~ N(NN(x(t)), R * R')

    For a Kalman Filter model, choose the observation network, NN(x), to be
    a one-layer network with a linear output. The latent state has
    dimensionality n (parameter "xDim") and observations have dimensionality m
    (parameter "yDim").

    Inputs:
    (See GenerativeModel abstract class definition for a list of standard
    parameters.)

    GenerativeParams  -  Dictionary of LDS parameters
                           * A     : [n x n] linear dynamics matrix; should
                                     have eigeninitial_values with magnitude
                                     strictly less than 1
                           * QChol : [n x n] square root of the innovation
                                     covariance Q
                           * Q0Chol: [n x n] square root of the innitial
                                     innovation covariance
                           * RChol : [n x 1] square root of the diagonal of the
                                     observation covariance
                           * x0    : [n x 1] mean of initial latent state
                           * NN_XtoY_Params:
                                   Dictionary with one field:
                                   - network: a lasagne network with input
                                   dimensionality n and output dimensionality m
    """
    def __init__(self, GenerativeParams, xDim, yDim, srng=None, nrng=None):

        super(LDS, self).__init__(GenerativeParams, xDim, yDim, srng, nrng)

        # parameters
        if 'A' in GenerativeParams:
            self.A = tf.Variable(initial_value=GenerativeParams['A']
                                 .astype(np.float32), name='A')
            # dynamics matrix
        else:
            # TBD:MAKE A BETTER WAY OF SAMPLING DEFAULT A
            self.A = tf.Variable(initial_value=.5*np.diag(np.ones(xDim)
                                 .astype(np.float32)), name='A')
            # dynamics matrix

        if 'QChol' in GenerativeParams:
            self.QChol = tf.Variable(initial_value=GenerativeParams['QChol']
                                     .astype(np.float32), name='QChol')
            # cholesky of innovation cov matrix
        else:
            self.QChol = tf.Variable(initial_value=(np.eye(xDim))
                                     .astype(np.float32), name='QChol')
            # cholesky of innovation cov matrix

        if 'Q0Chol' in GenerativeParams:
            self.Q0Chol = tf.Variable(initial_value=GenerativeParams['Q0Chol']
                                      .astype(np.float32), name='Q0Chol')
            # cholesky of starting distribution cov matrix
        else:
            self.Q0Chol = tf.Variable(initial_value=(np.eye(xDim))
                                      .astype(np.float32), name='Q0Chol')
            # cholesky of starting distribution cov matrix

        if 'RChol' in GenerativeParams:
            self.RChol = tf.Variable(initial_value=np.ndarray.flatten
                                     (GenerativeParams['RChol']
                                      .astype(np.float32)), name='RChol')
            # cholesky of observation noise cov matrix
        else:
            self.RChol = tf.Variable(initial_value=np.random.randn(yDim)
                                     .astype(np.float32)/10, name='RChol')
            # cholesky of observation noise cov matrix

        if 'x0' in GenerativeParams:
            self.x0 = tf.Variable(initial_value=GenerativeParams['x0']
                                  .astype(np.float32), name='x0')
            # set to zero for stationary distribution
        else:
            self.x0 = tf.Variable(initial_value=np.zeros((xDim,))
                                  .astype(np.float32), name='x0')
            # set to zero for stationary distribution

        if 'NN_XtoY_Params' in GenerativeParams:
            self.NN_XtoY = GenerativeParams['NN_XtoY_Params']['network']
        else:
            # Define a neural network that maps the latent state into the
            # output
            gen_nn = tf.contrib.keras.layers.Input((None, xDim))
            gen_nn_d = (tf.contrib.keras.layers.Dense(yDim,
                        activation="linear",
                        kernel_initializer=tf.orthogonal_initializer())
                        (gen_nn))
            self.NN_XtoY = tf.contrib.keras.models.Model(inputs=gen_nn,
                                                         outputs=gen_nn_d)

        # set to our lovely initial initial_values
        if 'C' in GenerativeParams:
            self.NN_XtoY.set_weights([GenerativeParams['C']
                                      .astype(np.float32),
                                      self.NN_XtoY.get_weights()[1]])
        if 'd' in GenerativeParams:
            self.NN_XtoY.set_weights([self.NN_XtoY.get_weights()[0],
                                      GenerativeParams['d']
                                      .astype(np.float32)])
        # we assume diagonal covariance (RChol is a vector)
        self.Rinv = 1./(self.RChol**2)
        # tf.matrix_inverse(tf.matmul(self.RChol ,T.transpose(self.RChol)))
        self.Lambda = tf.matrix_inverse(tf.matmul(self.QChol,
                                        tf.transpose(self.QChol)))
        self.Lambda0 = tf.matrix_inverse(tf.matmul(self.Q0Chol,
                                         tf.transpose(self.Q0Chol)))

        # Call the neural network output a rate, basically to keep things
        # consistent with the PLDS class
        self.rate = self.NN_XtoY(self.Xsamp)

    def sampleX(self, _N):
        _x0 = np.asarray(self.x0.eval(), dtype=np.float32)
        _Q0Chol = np.asarray(self.Q0Chol.eval(), dtype=np.float32)
        _QChol = np.asarray(self.QChol.eval(), dtype=np.float32)
        _A = np.asarray(self.A.eval(), dtype=np.float32)

        norm_samp = np.random.randn(_N, self.xDim).astype(np.float32)
        x_vals = np.zeros([_N, self.xDim]).astype(np.float32)

        x_vals[0] = _x0 + np.dot(norm_samp[0], _Q0Chol.T)

        for ii in range(_N-1):
            x_vals[ii+1] = x_vals[ii].dot(_A.T) + norm_samp[ii+1].dot(_QChol.T)

        return x_vals.astype(np.float32)

    def sampleY(self):
        """ Return a symbolic sample from the generative model. """
        return self.rate + tf.matmul(tf.random_normal((tf.shape(self.Xsamp)[0],
                                     self.yDim), seed=1234),
                                     tf.transpose(tf.diag(self.RChol)))

    def sampleXY(self, _N):
        """ Return numpy samples from the generative model. """
        X = self.sampleX(_N)
        nprand = np.random.randn(X.shape[0], self.yDim).astype(np.float32)
        _RChol = np.asarray(self.RChol.eval(), dtype=np.float32)
        Y = self.rate.eval({self.Xsamp: X}) + np.dot(nprand, np.diag(_RChol).T)
        return [X, Y]

    def getParams(self):
        return [self.A] + [self.QChol] + [self.Q0Chol] + [self.RChol]
        + [self.x0] + self.NN_XtoY.variables

    def evaluateLogDensity(self, X, Y):
        # Create a new graph which computes self.rate after replacing
        # self.Xsamp with X
        Ypred = tf.contrib.graph_editor.graph_replace(self.rate,
                                                      {self.Xsamp: X})
        resY = Y-Ypred
        self.resY = resY
        resX = X[1:]-tf.matmul(X[:(tf.shape(X)[0]-1)], tf.transpose(self.A))
        self.resX = resX
        self.resX0 = X[0]-self.x0

        LogDensity = -tf.reduce_sum(0.5*tf.matmul(tf.transpose(resY),
                                                  resY)*tf.diag(self.Rinv))

        LogDensity += -tf.reduce_sum(0.5*tf.matmul(tf.transpose(resX),
                                                   resX)*self.Lambda)

        LogDensity += -0.5*tf.matmul(tf.matmul(tf.reshape(self.resX0,
                                                          [1, -1]),
                                               self.Lambda0),
                                     tf.reshape(self.resX0, [-1, 1]))

        LogDensity += (0.5*tf.reduce_sum(tf.log(self.Rinv))
                       * (tf.shape(Y, out_type=tf.float32)[0]))

        LogDensity += (0.5*tf.log(tf.matrix_determinant(self.Lambda))
                       * (tf.shape(Y, out_type=tf.float32)[0]-1))

        LogDensity += 0.5*tf.log(tf.matrix_determinant(self.Lambda0))

        LogDensity += (-0.5*(self.xDim + self.yDim)*np.log(2*np.pi)
                       * tf.shape(Y, out_type=tf.float32)[0])

        return LogDensity

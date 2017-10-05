import tensorflow as tf
from tensorflow.contrib.keras import layers, models
from tf_gbds.CGAN import WGAN
import numpy as np


class GenerativeModel(object):
    """
    Interface class for generative time-series models
    """
    def __init__(self, GenerativeParams, xDim, yDim, nrng=None):

        # input variable referencing top-down or external input

        self.xDim = xDim
        self.yDim = yDim

        self.nrng = nrng

        # internal RV for generating sample
        self.Xsamp = tf.placeholder(tf.float32, shape=[None, xDim],
                                    name="Xsamp")

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
    def __init__(self, GenerativeParams, xDim, yDim, nrng=None):

        super(LDS, self).__init__(GenerativeParams, xDim, yDim, nrng)

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
            gen_nn = layers.Input(batch_shape=(None, xDim))
            gen_nn_d = (layers.Dense(yDim,
                        activation="linear",
                        kernel_initializer=tf.orthogonal_initializer())
                        (gen_nn))
            self.NN_XtoY = models.Model(inputs=gen_nn, outputs=gen_nn_d)

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
        # Return a symbolic sample from the generative model.
        return self.rate + tf.matmul(tf.random_normal((tf.shape(self.Xsamp)[0],
                                     self.yDim), seed=1234),
                                     tf.transpose(tf.diag(self.RChol)))

    def sampleXY(self, _N):
        # Return numpy samples from the generative model.
        X = self.sampleX(_N)
        nprand = np.random.randn(X.shape[0], self.yDim).astype(np.float32)
        _RChol = np.asarray(self.RChol.eval(), dtype=np.float32)
        Y = self.rate.eval({self.Xsamp: X}) + np.dot(nprand, np.diag(_RChol).T)
        return [X, Y]

    def getParams(self):
        return ([self.A] + [self.QChol] + [self.Q0Chol] + [self.RChol] +
                [self.x0] + self.NN_XtoY.variables)

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

        LogDensity += (0.5*tf.reduce_sum(tf.log(self.Rinv)) *
                       (tf.cast(tf.shape(Y)[0], tf.float32)))

        LogDensity += (0.5*tf.log(tf.matrix_determinant(self.Lambda)) *
                       (tf.cast(tf.shape(Y)[0]-1, tf.float32)))

        LogDensity += 0.5*tf.log(tf.matrix_determinant(self.Lambda0))

        LogDensity += (-0.5*(self.xDim + self.yDim)*np.log(2*np.pi) *
                       tf.cast(tf.shape(Y)[0], tf.float32))

        return LogDensity


def logsumexp(x, axis=None):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    return (tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=axis, keep_dims=True))
            + x_max)


class GBDS(GenerativeModel):
    """
    Goal-Based Dynamical System

    Inputs:
    - GenerativeParams: A dictionary of parameters for the model
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
        - NN_postJ_sigma: Neural network that parametrizes the covariance of
                          the posterior of J (i.e. mu and sigma), conditioned
                          on goals
        - yCols: Columns of Y this agent corresponds to. Used to index columns
                 in real data to compare against generated data.
        - vel: Maximum velocity of each dimension in yCols.
    - yDim: Number of dimensions for this agent
    - yDim_in: Number of total dimensions in the data
    - srng: Theano symbolic random number generator (theano RandomStreams
            object)
    - nrng: Numpy random number generator
    """
    def __init__(self, GenerativeParams, yDim, yDim_in, nrng=None):
        super(GBDS, self).__init__(GenerativeParams, yDim, yDim, nrng)
        self.yDim_in = yDim_in  # dimension of observation input
        # function that calculates states from positions
        self.get_states = GenerativeParams['get_states']

        # GMM networks
        self.GMM_k = GenerativeParams['GMM_k']  # number of GMM components
        self.GMM_net = GenerativeParams['GMM_net']
        output = (layers.Lambda(lambda x: x[:, :yDim *
                                self.GMM_k])(self.GMM_net.output))
        self.GMM_mu = models.Model(self.GMM_net.input, output)
        output = (layers.Activation('softplus')(
            layers.Lambda(lambda x: x[:, yDim * self.GMM_k:2 * yDim *
                          self.GMM_k])(self.GMM_net.output)))
        self.GMM_lambda = models.Model(self.GMM_net.input, output)
        output = (layers.Activation('softmax')(
            layers.Lambda(lambda x: x[:, 2 * yDim *
                          self.GMM_k:])(self.GMM_net.output)))
        self.GMM_w = models.Model(self.GMM_net.input, output)

        # penalty on epsilon (noise on control signal)

        if 'pen_eps' in GenerativeParams:
            self.pen_eps = GenerativeParams['pen_eps']
        else:
            self.pen_eps = None

        # penalty on sigma (noise on goal state)
        if 'pen_sigma' in GenerativeParams:
            self.pen_sigma = GenerativeParams['pen_sigma']
        else:
            self.pen_sigma = None

        # penalties on goal state passing boundaries
        if 'pen_g' in GenerativeParams:
            self.pen_g = GenerativeParams['pen_g']
        else:
            self.pen_g = (None, None)

        # corresponding boundaries for pen_g
        if 'bounds_g' in GenerativeParams:
            self.bounds_g = GenerativeParams['bounds_g']
        else:
            self.bounds_g = (1.0, 1.5)
        # which dimensions of Y to predict
        self.yCols = GenerativeParams['yCols']

        # velocity for each observation dimension (of all agents)
        self.all_vel = tf.constant(GenerativeParams['all_vel'], tf.float32)
        # velocity for each observation dimension (of this agent)
        self.vel = tf.constant(GenerativeParams['vel'], tf.float32)

        # coefficients for PID controller (one for each dimension)
        # https://en.wikipedia.org/wiki/PID_controller#Discrete_implementation
        unc_Kp = tf.Variable(initial_value=np.zeros((self.yDim, 1)),
                             name='unc_Kp', dtype=tf.float32)
        unc_Ki = tf.Variable(initial_value=np.zeros((self.yDim, 1)),
                             name='unc_Ki', dtype=np.float32)
        unc_Kd = tf.Variable(initial_value=np.zeros((self.yDim, 1)),
                             name='unc_Kd', dtype=np.float32)

        # create list of PID controller parameters for easy access in
        # getParams
        # self.PID_params = [unc_Kp]
        self.PID_params = [unc_Kp, unc_Ki, unc_Kd]

        # constrain PID controller parameters to be positive
        self.Kp = tf.nn.softplus(unc_Kp)
        # self.Ki = unc_Ki
        self.Ki = tf.nn.softplus(unc_Ki)
        # self.Kd = unc_Kd
        self.Kd = tf.nn.softplus(unc_Kd)

        # calculate coefficients to be placed in convolutional filter
        t_coeff = self.Kp + self.Ki + self.Kd
        t1_coeff = -self.Kp - 2 * self.Kd
        t2_coeff = self.Kd

        # concatenate coefficients into filter
        self.L = tf.concat([t_coeff, t1_coeff, t2_coeff], axis=1)

        # noise coefficient on goal states
        self.unc_sigma = tf.Variable(initial_value=-7 * np.ones((1,
                                                                 self.yDim)),
                                     name='unc_sigma', dtype=tf.float32)
        self.sigma = tf.nn.softplus(self.unc_sigma)

        # noise coefficient on control signals
        self.unc_eps = tf.Variable(initial_value=-11 * np.ones((1,
                                                                self.yDim)),
                                   name='unc_eps', dtype=tf.float32)
        self.eps = tf.nn.softplus(self.unc_eps)

    def sample_GMM(self, states):
        """
        Sample from GMM based on highest weight component
        """
        mu = tf.reshape(self.GMM_mu(states), [-1, self.GMM_k, self.yDim])
        lmbda = tf.reshape(self.GMM_lambda(states),
                           [-1, self.GMM_k, self.yDim])
        all_w = self.GMM_w(states)

        def select_components(acc, inputs):
            sub_mu, sub_lmbda, w = inputs
            a = tf.range(self.GMM_k)
            p = tf.multinomial(tf.expand_dims(tf.reshape(w, [-1]), -1), 1)
            component = a[tf.cast(p[0][0], tf.int32)]

            return sub_mu[component, :], sub_lmbda[component, :]

        (mu_k, lmbda_k) = (tf.scan(select_components, [mu, lmbda, all_w],
                                   initializer=(tf.zeros([self.yDim]),
                                   tf.zeros([self.yDim]))))
        updates = {}

        return (mu, lmbda, all_w, mu_k, lmbda_k), updates

    def get_preds(self, Y, training=False, post_g=None,
                  gen_g=None, extra_conds=None):
        """
        Return the predicted next J, g, U, and Y for each point in Y.

        For training: provide post_g, sample from the posterior,
                      which is used to calculate the ELBO
        For generating new data: provide gen_g, the generated goal states up to
                                 the current timepoint
        """
        if training and post_g is None:
            raise Exception(
                "Must provide sample of g from posterior during training")
        # get states from position
        states = tf.constant(self.get_states(Y, max_vel=self.all_vel),
                             tf.float32)
        if extra_conds is not None:
            states = tf.concat([states, extra_conds], axis=1)
        (all_mu, all_lmbda, all_w, mu_k,
            lmbda_k), updates = self.sample_GMM(states)
        # Draw next goals based on force
        if post_g is not None:  # Calculate next goals from posterior
            next_g = ((tf.reshape(post_g[:-1], [-1, 1, self.yDim]) +
                       all_mu * all_lmbda) / (1 + all_lmbda))
        elif gen_g is not None:  # Generate next goals
            # Get external force from GMM
            goal = ((gen_g[(-1,)] + lmbda_k[(-1,)] * mu_k[(-1,)]) /
                    (1 + lmbda_k[(-1,)]))
            var = self.sigma**2 / (1 + lmbda_k[(-1,)])
            goal += tf.random.normal(goal.shape) * tf.sqrt(var)
            next_g = tf.concat([gen_g[1:], goal], axis=0)
        else:
            raise Exception("Goal states must be provided " +
                            "(either posterior or generated)")
        # PID Controller for next control point
        if post_g is not None:  # calculate error from posterior goals
            error = post_g[1:] - tf.gather(Y, self.yCols, axis=1)
        else:  # calculate error from generated goals
            error = next_g - tf.gather(Y, self.yCols, axis=1)

        # Assume control starts at zero
        Uprev = tf.concat([tf.zeros([1, self.yDim]),
                           tf.atanh((tf.gather(Y, self.yCols, axis=1)[1:] -
                                     tf.gather(Y, self.yCols, axis=1)[:-1]) /
                                    tf.reshape(self.vel, [1, self.yDim]))],
                          axis=0)
        Udiff = []
        for i in range(self.yDim):
            # get current error signal and corresponding filter
            signal = error[:, i]
            filt = tf.reshape(self.L[i], [-1, 1, 1, 1])
            # zero pad beginning
            signal = tf.reshape(tf.concat([tf.zeros(2), signal], axis=0),
                                [1, -1, 1, 1])
            res = tf.nn.conv2d(signal, filt, strides=[1, 1, 1, 1],
                               padding="VALID")
            res = tf.reshape(res, [-1, 1])
            Udiff.append(res)
        if len(Udiff) > 1:
            Udiff = tf.concat([*Udiff], axis=1)
        else:
            Udiff = Udiff[0]
        if post_g is None:  # Add control signal noise to generated data
            Udiff += self.eps * tf.random_normal(Udiff.shape)
        Upred = Uprev + Udiff
        # get predicted Y
        Ypred = (tf.gather(Y, self.yCols, axis=1) +
                 tf.reshape(self.vel, [1, self.yDim]) * tf.tanh(Upred))

        return (all_mu, all_lmbda, all_w, next_g, Upred, Ypred), updates

    def evaluateLogDensity(self, g, Y):
        '''
        Return a theano function that evaluates the log-density of the
        GenerativeModel.

        g: Goal state time series (sample from the recognition model)
        Y: Time series of positions
        '''
        # Calculate real control signal
        U_true = tf.atanh((tf.gather(Y, self.yCols, axis=1)[1:] -
                           tf.gather(Y, self.yCols, axis=1)[:-1]) /
                          tf.reshape(self.vel, [1, self.yDim]))
        # Get predictions for next timestep (at each timestep except for last)
        # disregard last timestep bc we don't know the next value, thus, we
        # can't calculate the error
        (all_mu, all_lmbda, all_w, g_pred, Upred,
            Ypred), updates = self.get_preds(Y[:-1], training=True, post_g=g)
        # calculate loss on control signal
        resU = U_true - Upred
        LogDensity = -tf.reduce_sum(resU**2 / (2 * self.eps**2))
        LogDensity -= (0.5 * tf.log(2 * np.pi) +
                       tf.reduce_sum(tf.log(self.eps)))

        # calculate GMM loss
        w_brdcst = tf.reshape(all_w, [-1, self.GMM_k, 1])
        gmm_res_g = tf.reshape(g[1:], [-1, 1, self.yDim]) - g_pred
        gmm_term = (tf.log(w_brdcst + 1e-8) - ((1 + all_lmbda) /
                    (2 * tf.reshape(self.sigma,
                                    [1, 1, -1])**2)) * gmm_res_g**2)
        gmm_term += (0.5 * tf.log(1 + all_lmbda) - 0.5 * tf.log(2 * np.pi) -
                     tf.log(tf.reshape(self.sigma, [1, 1, -1])))
        LogDensity += tf.reduce_sum(logsumexp(tf.reduce_sum(gmm_term, axis=2),
                                    axis=1))

        # linear penalty on goal state escaping game space
        if self.pen_g[0] is not None:
            LogDensity -= (self.pen_g[0] * tf.reduce_sum(
                tf.nn.relu(g_pred - self.bounds_g[0])))
            LogDensity -= (self.pen_g[0] * tf.reduce_sum(
                tf.nn.relu(-g_pred - self.bounds_g[0])))
        if self.pen_g[1] is not None:
            LogDensity -= (self.pen_g[1] * tf.reduce_sum(
                tf.nn.relu(g_pred - self.bounds_g[1])))
            LogDensity -= (self.pen_g[1] * tf.reduce_sum(
                tf.nn.relu(-g_pred - self.bounds_g[1])))

        # penalty on eps
        if self.pen_eps is not None:
            LogDensity -= self.pen_eps * tf.reduce_sum(self.unc_eps)

        # # penalty on sigma
        # if self.pen_sigma is not None:
        #     LogDensity -= self.pen_sigma * tf.reduce_sum(self.unc_sigma)

        return LogDensity, updates

    def init_GAN(self, *args, **kwargs):
        """
        Initialize Generative Adversarial Network that generates
        initial goal state, g_0, from random noise
        Look at CGAN.py for initialization parameters.
        This function exists so that a control model can be trained, and
        then, several GANs can be trained using that control model.
        """
        # kwargs['srng'] = self.srng
        self.GAN_g0 = WGAN(*args, **kwargs)

    def evaluateGANLoss(self, post_g0, mode='D'):
        """
        Evaluate loss of GAN
        Mode is D for discriminator, G for generator
        """
        if self.GAN_g0 is None:
            raise Exception("Must initiate GAN before calling")
        # Get external force from CGAN
        gen_g0 = self.GAN_g0.get_generated_data(post_g0.shape[0],
                                                training=True)
        if mode == 'D':
            return self.GAN_g0.get_discr_cost(post_g0, gen_g0)
        elif mode == 'G':
            return self.GAN_g0.get_gen_cost(gen_g0)
        else:
            raise Exception("Invalid mode. Provide 'G' for generator loss " +
                            "or 'D' for discriminator loss.")

    def getParams(self):
        '''
        Return the learnable parameters of the model
        '''
        rets = self.GMM_net.variables
        rets += self.PID_params + [self.unc_eps]  # + [self.unc_sigma]
        return rets


class PLDS(LDS):
    """
    Gaussian linear dynamical system with Poisson count observations. Inherits
    Gaussian linear dynamics sampling code from the LDS; implements a Poisson
    density evaluation for discrete (count) data.
    """
    def __init__(self, GenerativeParams, xDim, yDim, nrng=None):
        # The LDS class expects "RChol" for Gaussian observations - we just
        # pass a dummy
        GenerativeParams['RChol'] = np.ones(1)
        super(PLDS, self).__init__(GenerativeParams, xDim, yDim, nrng)

        # Currently we emulate a PLDS by having an exponential output
        # nonlinearity. Next step will be to generalize this to more flexible
        # output nonlinearities...
        if GenerativeParams['output_nlin'] == 'exponential':
            self.rate = tf.exp(self.NN_XtoY(self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'sigmoid':
            self.rate = tf.nn.sigmoid(self.NN_XtoY(self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'softplus':
            self.rate = tf.nn.softplus(self.NN_XtoY(self.Xsamp))
        else:
            raise Exception('Unknown output nonlinearity specification!')

    def getParams(self):
        return ([self.A] + [self.QChol] + [self.Q0Chol] + [self.x0] +
                self.NN_XtoY.variables)

    def sampleY(self):
        # Return a symbolic sample from the generative model.
        return tf.random_poisson(lam=self.rate, shape=tf.shape(self.rate),
                                 seed=1234)

    def sampleXY(self, _N):
        # Return real-valued (numpy) samples from the generative model.
        X = self.sampleX(_N)
        Y = np.random.poisson(lam=self.rate.eval({self.Xsamp: X}))
        return [X, Y]

    def evaluateLogDensity(self, X, Y):
        # This is the log density of the generative model (*not* negated)
        Ypred = tf.contrib.graph_editor.graph_replace(self.rate,
                                                      {self.Xsamp: X})
        resY = Y-Ypred
        self.resY = resY
        resX = X[1:]-tf.matmul(X[:-1], tf.transpose(self.A))
        self.resX = resX
        resX0 = X[0]-self.x0
        self.resX0 = resX0
        LatentDensity = (-0.5*tf.matmul(tf.matmul(tf.expand_dims(
          resX0, 0), self.Lambda0), tf.transpose(tf.expand_dims(resX0, 0))) -
                         0.5*tf.reduce_sum(resX*tf.matmul(resX,
                                                          self.Lambda)) +
                         0.5*tf.log(tf.matrix_determinant(self.Lambda)) *
                         (tf.cast(tf.shape(Y)[0]-1, tf.float32)) +
                         0.5*tf.log(tf.matrix_determinant(self.Lambda0)) -
                         0.5*(self.xDim)*np.log(2*np.pi) *
                         (tf.cast(tf.shape(Y)[0], tf.float32)))
        PoisDensity = tf.reduce_sum(Y*tf.log(Ypred)-Ypred-tf.lgamma(Y+1))
        LogDensity = LatentDensity + PoisDensity
        return LogDensity

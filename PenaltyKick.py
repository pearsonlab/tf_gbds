import tensorflow as tf
import numpy as np
from tf_gbds.GenerativeModel import GBDS
from tf_gbds.RecognitionModel import SmoothingPastLDSTimeSeries


class SGVB_GBDS():  # (Trainable):
    '''
    This class fits a model to PenaltyKick data.

    Inputs:
    gen_params_ball       - Dictionary of parameters that define the chosen
                            GenerativeModel
                            for the ball/shooter in penaltyshot
                            Look inside the class for details on what to
                            include.
    gen_params_goalie     - Dictionary of parameters that define the chosen
                            GenerativeModel
                            for the goalie in penaltyshot
                            Look inside the class for details on what to
                            include.
    yCols_ball       - Dimensions of Y that correspond to ball coordinates
    yCols_goalie     - Dimenstions of Y that correspond to goalie coordinates
    gen_params       - Dictionary of parameters that define the chosen
                       RecognitionModel.
                       Look inside the class for details on what to include.
    ntrials          - Number of trials in the training dataset

    --------------------------------------------------------------------------

    The SGVB ("Stochastic Gradient Variational Bayes") inference technique is
    described in the following publications:
    * Auto-Encoding Variational Bayes
           - Kingma, Welling (ICLR, 2014)
    * Stochastic backpropagation and approximate inference in deep generative
      models.
           - Rezende et al (ICML, 2014)
    * Doubly stochastic variational bayes for non-conjugate inference.
           - Titsias and Lazaro-Gredilla (ICML, 2014)
    '''
    def __init__(self, gen_params_ball, gen_params_goalie, yCols_ball,
                 yCols_goalie, rec_params, ntrials):
        # instantiate rng's
        self.nrng = np.random.RandomState(124)

        # actual model parameters
        # symbolic variables for VB training
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, shape=(None, None), name='X')
            self.Y = tf.placeholder(tf.float32, shape=(None, None), name='Y')

        # symbolic variables for CGAN training
        self.J = tf.placeholder(tf.float32, shape=(None, None), name='J')
        self.s = tf.placeholder(tf.float32, shape=(None, None), name='s')

        # symbolic variables for GAN training
        self.g0 = tf.placeholder(tf.float32, shape=(None, None), name='g0')

        with tf.name_scope('columns_by_agent'):
            self.yCols_goalie = yCols_goalie
            self.yCols_ball = yCols_ball
            self.yDim_goalie = len(self.yCols_goalie)
            self.yDim_ball = len(self.yCols_ball)
            self.yDim = tf.shape(self.Y)[1]
            self.xDim = self.yDim

        # instantiate our prior and recognition models
        with tf.name_scope('init_model'):
            with tf.name_scope('mrec'):
                self.mrec = SmoothingPastLDSTimeSeries(rec_params, self.Y, self.xDim,
                                                       self.yDim, ntrials,
                                                       self.nrng)
            with tf.name_scope('mprior'):
                with tf.name_scope('mprior_goalie'):
                    self.mprior_goalie = GBDS(gen_params_goalie,
                                              self.yDim_goalie, self.yDim,
                                              nrng=self.nrng)
                with tf.name_scope('mprior_ball'):
                    self.mprior_ball = GBDS(gen_params_ball,
                                            self.yDim_ball, self.yDim,
                                            nrng=self.nrng)

        with tf.name_scope('default_training_mode'):
            self.isTrainingGenerativeModel = True
            self.isTrainingRecognitionModel = True
            self.isTrainingCGANGenerator = False
            self.isTrainingCGANDiscriminator = False
            self.isTrainingGANGenerator = False
            self.isTrainingGANDiscriminator = False

    def getParams(self):
        '''
        Return Generative and Recognition Model parameters that are currently
        being trained.
        '''
        params = []
        if self.isTrainingRecognitionModel:
            params += self.mrec.getParams()
        if self.isTrainingGenerativeModel:
            params += self.mprior_goalie.getParams()
            params += self.mprior_ball.getParams()
        if self.isTrainingCGANGenerator:
            params += self.mprior_ball.CGAN_J.get_gen_params()
            params += self.mprior_goalie.CGAN_J.get_gen_params()
        if self.isTrainingCGANDiscriminator:
            params += self.mprior_ball.CGAN_J.get_discr_params()
            params += self.mprior_goalie.CGAN_J.get_discr_params()
        if self.isTrainingGANGenerator:
            params += self.mprior_ball.GAN_g0.get_gen_params()
            params += self.mprior_goalie.GAN_g0.get_gen_params()
        if self.isTrainingGANDiscriminator:
            params += self.mprior_ball.GAN_g0.get_discr_params()
            params += self.mprior_goalie.GAN_g0.get_discr_params()

        return params

    def set_training_mode(self, mode):
        '''
        Set training flags for appropriate mode.
        Options for mode are as follows:
        'CTRL': Trains the generative and recognition control model jointly
        'CGAN_G': Trains the CGAN generator
        'CGAN_D': Trains the CGAN discriminator
        'GAN_G': Trains the GAN generator
        'GAN_D': Trains the GAN discriminator
        '''
        if mode == 'CTRL':
            self.isTrainingGenerativeModel = True
            self.isTrainingRecognitionModel = True
            self.isTrainingCGANGenerator = False
            self.isTrainingCGANDiscriminator = False
            self.isTrainingGANGenerator = False
            self.isTrainingGANDiscriminator = False
        elif mode == 'CGAN_G':
            self.isTrainingGenerativeModel = False
            self.isTrainingRecognitionModel = False
            self.isTrainingCGANGenerator = True
            self.isTrainingCGANDiscriminator = False
            self.isTrainingGANGenerator = False
            self.isTrainingGANDiscriminator = False
        elif mode == 'CGAN_D':
            self.isTrainingGenerativeModel = False
            self.isTrainingRecognitionModel = False
            self.isTrainingCGANGenerator = False
            self.isTrainingCGANDiscriminator = True
            self.isTrainingGANGenerator = False
            self.isTrainingGANDiscriminator = False
        elif mode == 'GAN_G':
            self.isTrainingGenerativeModel = False
            self.isTrainingRecognitionModel = False
            self.isTrainingCGANGenerator = False
            self.isTrainingCGANDiscriminator = False
            self.isTrainingGANGenerator = True
            self.isTrainingGANDiscriminator = False
        elif mode == 'GAN_D':
            self.isTrainingGenerativeModel = False
            self.isTrainingRecognitionModel = False
            self.isTrainingCGANGenerator = False
            self.isTrainingCGANDiscriminator = False
            self.isTrainingGANGenerator = False
            self.isTrainingGANDiscriminator = True

    def cost(self):
        '''
        Compute a one-sample approximation the ELBO (lower bound on marginal
        likelihood), normalized by batch size (length of Y in first dimension).
        '''
        JCols_goalie = range(self.yDim_goalie * 2)
        JCols_ball = range(self.yDim_goalie * 2,
                           self.yDim_goalie * 2 + self.yDim_ball * 2)
        cost = 0
        with tf.name_scope('goal_samples'):
            q = tf.squeeze(self.mrec.samples, 2, name='q')
        with tf.name_scope('gen_cost'):
            if self.isTrainingGenerativeModel or self.isTrainingRecognitionModel:
                with tf.name_scope('gen_goalie_cost'):
                    cost += self.mprior_goalie.evaluateLogDensity(
                        tf.gather(q, self.yCols_goalie, axis=1), self.Y)
                with tf.name_scope('gen_ball_cost'):
                    cost += self.mprior_ball.evaluateLogDensity(
                        tf.gather(q, self.yCols_ball, axis=1), self.Y)
        with tf.name_scope('rec_cost'):
            if self.isTrainingRecognitionModel:
                cost += self.mrec.evalEntropy()
        with tf.name_scope('GAN_loss'):
            if self.isTrainingCGANGenerator:
                cost += self.mprior_ball.evaluateCGANLoss(self.J[:, JCols_ball],
                                                          self.s, mode='G')
                cost += self.mprior_goalie.evaluateCGANLoss(
                    self.J[:, JCols_goalie], self.s, mode='G')
            if self.isTrainingCGANDiscriminator:
                cost += self.mprior_ball.evaluateCGANLoss(self.J[:, JCols_ball],
                                                          self.s, mode='D')
                cost += self.mprior_goalie.evaluateCGANLoss(
                    self.J[:, JCols_goalie], self.s, mode='D')
            if self.isTrainingGANGenerator:
                cost += self.mprior_ball.evaluateGANLoss(
                    tf.gather(self.g0, self.yCols_ball, axis=1), mode='G')
                cost += self.mprior_goalie.evaluateGANLoss(
                    tf.gather(self.g0, self.yCols_goalie, axis=1), mode='G')
            if self.isTrainingGANDiscriminator:
                cost += self.mprior_ball.evaluateGANLoss(
                    tf.gather(self.g0, self.yCols_ball, axis=1), mode='D')
                cost += self.mprior_goalie.evaluateGANLoss(
                    tf.gather(self.g0, self.yCols_goalie, axis=1), mode='D')
        if self.isTrainingGenerativeModel or self.isTrainingRecognitionModel:
            return cost / tf.cast(tf.shape(self.Y)[0], tf.float32)
        else:  # training GAN
            return cost

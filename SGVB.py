import tensorflow as tf
import numpy as np


class SGVB():
    """
    This class defines a variational inference method based on a one-sample
    approximation to the ELBO, given specified Generative and Recognition
    Models (the recognition model is also known as an "approximate posterior").
    For algorithm details, see references below.

    Inputs:
    opt_params (optional) - Dictionary of parameters used during optimziation
                            (TODO)
    gen_params            - Dictionary of parameters that define the chosen
                            GenerativeModel
    GEN_MODEL             - A class that inhereits from the GenerativeModel
                            abstract class
    rec_params            - Dictionary of parameters that define the chosen
                            RecognitionModel
    REC_MODEL             - A class that inhereits from the RecognitionModel
                            abstract class
    xDim                  - Integer that specifies the dimensionality of the
                            latent space
    yDim                  - Integer that specifies the dimensionality of the
                            observations

    --------------------------------------------------------------------------
    This code is a reference implementation of the algorithm described in:
    * Black box variational inference for state space models
           - Archer et al (arxiv preprint, 2015)
             [http://arxiv.org/abs/1511.07367]

    The SGVB ("Stochastic Gradient Variational Bayes") inference technique is
    described
    in the following publications:
    * Auto-Encoding Variational Bayes
           - Kingma, Welling (ICLR, 2014)
    * Stochastic backpropagation and approximate inference in deep generative
      models.
           - Rezende et al (ICML, 2014)
    * Doubly stochastic variational bayes for non-conjugate inference.
           - Titsias and Lazaro-Gredilla (ICML, 2014)
    """
    def __init__(self,
                 gen_params,  # dictionary of generative model parameters
                 GEN_MODEL,   # class that inherits from GenerativeModel
                 rec_params,  # dictionary of approximate posterior
                              # ("recognition model") parameters
                 REC_MODEL,   # class that inherits from RecognitionModel
                 xDim=2,      # dimensionality of latent state
                 yDim=2       # dimensionality of observations
                 ):

        # instantiate rng's
        self.srng = None
        self.nrng = np.random.RandomState(124)

        # ---------------------------------------------------------
        # actual model parameters
        # self.X, self.Y = T.matrices('X','Y')   # symbolic variables for the
        # data

        self.X = tf.placeholder(tf.float32,
                                shape=(None, xDim), name='X')
        self.Y = tf.placeholder(tf.float32,
                                shape=(None, yDim), name='Y')

        self.xDim = xDim
        self.yDim = yDim

        # instantiate our prior & recognition models
        self.mrec = REC_MODEL(rec_params, self.Y, self.xDim, self.yDim,
                              self.srng, self.nrng)
        self.mprior = GEN_MODEL(gen_params, self.xDim, self.yDim,
                                srng=self.srng, nrng=self.nrng)

        self.isTrainingRecognitionModel = True
        self.isTrainingGenerativeModel = True

    def getParams(self):
        """
        Return Generative and Recognition Model parameters that are currently
        being trained.
        """
        params = []
        if self.isTrainingRecognitionModel:
            params = params + self.mrec.getParams()
        if self.isTrainingGenerativeModel:
            params = params + self.mprior.getParams()
        return params

    def EnableRecognitionModelTraining(self):
        """
        Enable training of RecognitionModel parameters.
        """
        self.isTrainingRecognitionModel = True
        self.mrec.setTrainingMode()

    def DisableRecognitionModelTraining(self):
        """
        Disable training of RecognitionModel parameters.
        """
        self.isTrainingRecognitionModel = False
        self.mrec.setTestMode()

    def EnableGenerativeModelTraining(self):
        """
        Enable training of GenerativeModel parameters.
        """
        self.isTrainingGenerativeModel = True
        print('Enable switching training/test mode' +
              ' in generative model class!\n')

    def DisableGenerativeModelTraining(self):
        """
        Disable training of GenerativeModel parameters.
        """
        self.isTrainingGenerativeModel = False
        print('Enable switching training/test mode' +
              ' in generative model class!\n')

    def cost(self):
        """
        Compute a one-sample approximation the ELBO (lower bound on marginal
        likelihood), normalized by batch size (length of Y in first dimension).
        """
        q = self.mrec.getSample()

        theentropy = self.mrec.evalEntropy()

        thelik = self.mprior.evaluateLogDensity(tf.squeeze(q, -1), self.Y)

        thecost = thelik + theentropy

        return thecost/tf.cast(tf.shape(self.Y)[0], dtype=tf.float32)

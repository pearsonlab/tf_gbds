import numpy as np
import tensorflow as tf
import numpy.testing as npt
import tf_gbds.SGVB as S
import tf_gbds.GenerativeModel as G
import tf_gbds.RecognitionModel as R
from tensorflow.contrib.keras import layers as keras_layers
from tensorflow.contrib.keras import models


def test_SGVB():

    xDim = 2
    yDim = 5

    mu_nn = keras_layers.Input((None, yDim))
    mu_nn_d = (keras_layers.Dense(xDim, activation="linear",
               kernel_initializer=tf.orthogonal_initializer())(mu_nn))
    NN_Mu = models.Model(inputs=mu_nn, outputs=mu_nn_d)

    lambda_nn = keras_layers.Input((None, yDim))
    lambda_nn_d = (keras_layers.Dense(xDim*xDim, activation="linear",
                   kernel_initializer=tf.orthogonal_initializer())(lambda_nn))
    NN_Lambda = models.Model(inputs=lambda_nn, outputs=lambda_nn_d)

    lambdax_nn = keras_layers.Input((None, yDim))
    lambdax_nn_d = (keras_layers.Dense(xDim*xDim, activation="linear",
                    kernel_initializer=tf.orthogonal_initializer())
                    (lambdax_nn))
    NN_LambdaX = models.Model(inputs=lambdax_nn, outputs=lambdax_nn_d)

    A = .5*np.diag(np.ones(xDim))
    QinvChol = np.eye(xDim)
    Q0invChol = np.eye(xDim)

    RecognitionParams = ({'NN_Mu': {'network': NN_Mu},
                          'NN_Lambda': {'network': NN_Lambda},
                          'NN_LambdaX': {'network': NN_LambdaX},
                          'A': A,
                          'QinvChol': QinvChol,
                          'Q0invChol': Q0invChol})

    sg = S.SGVB(gen_params={}, GEN_MODEL=G.LDS, rec_params=RecognitionParams,
                REC_MODEL=R.SmoothingLDSTimeSeries, xDim=2, yDim=5)
    assert isinstance(sg.X, tf.Tensor)
    assert isinstance(sg.Y, tf.Tensor)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = np.reshape(np.arange(0, 20), [10, 2])
        y = np.reshape(np.arange(0, 50), [10, 5])
        q = sg.mrec.samples.eval(feed_dict={sg.X: x, sg.Y: y})
        q_squeeze = tf.squeeze(tf.constant(q, dtype=tf.float32), -1)
        theentropy = sg.mrec.evalEntropy().eval(feed_dict={sg.X: x, sg.Y: y})
        thelik = sg.mprior.evaluateLogDensity(q_squeeze,
                                              sg.Y).eval(feed_dict={sg.X: x,
                                                                    sg.Y: y})
        thecost = (thelik + theentropy)
        cost = thecost/y.shape[0]

        theentropy = sg.mrec.evalEntropy()
        thelik = sg.mprior.evaluateLogDensity(q_squeeze, sg.Y)
        thecost = thelik + theentropy
        cost_tf = thecost/tf.cast(tf.shape(sg.Y)[0], dtype=tf.float32)

        npt.assert_allclose(cost, cost_tf.eval(feed_dict={sg.X: x, sg.Y: y}),
                            atol=1e-5, rtol=1e-5)

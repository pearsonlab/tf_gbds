import numpy as np
import tensorflow as tf
import numpy.testing as npt
from tensorflow.contrib.keras import layers, models
import tf_gbds.RecognitionModel as R
import tf_gbds.lib.sym_blk_tridiag_inv as sym


def test_recognition_model():
    rm = R.RecognitionModel(None, 5, 10)

    assert rm.xDim == 5
    assert rm.yDim == 10


def test_smoothing_LDS_time_series_recognition():
    xDim = 2
    yDim = 5
    Input = tf.placeholder(tf.float32, shape=(None, yDim), name='Input')
    T = 10

    mu_nn = layers.Input((None, yDim))
    mu_nn_d = (layers.Dense(xDim, activation="linear",
               kernel_initializer=tf.orthogonal_initializer())(mu_nn))
    NN_Mu = models.Model(inputs=mu_nn, outputs=mu_nn_d)

    lambda_nn = layers.Input((None, yDim))
    lambda_nn_d = (layers.Dense(xDim*xDim, activation="linear",
                   kernel_initializer=tf.orthogonal_initializer())(lambda_nn))
    NN_Lambda = models.Model(inputs=lambda_nn, outputs=lambda_nn_d)

    lambdax_nn = layers.Input((None, yDim))
    lambdax_nn_d = (layers.Dense(xDim*xDim, activation="linear",
                    kernel_initializer=tf.orthogonal_initializer())
                    (lambdax_nn))
    NN_LambdaX = models.Model(inputs=lambdax_nn, outputs=lambdax_nn_d)

    Data = np.zeros([10, 5])

    A = .5*np.diag(np.ones(xDim))
    QinvChol = np.eye(xDim)
    Q0invChol = np.eye(xDim)

    RecognitionParams = ({'NN_Mu': {'network': NN_Mu},
                          'NN_Lambda': {'network': NN_Lambda},
                          'NN_LambdaX': {'network': NN_LambdaX},
                          'A': A,
                          'QinvChol': QinvChol,
                          'Q0invChol': Q0invChol})

    Qinv = np.dot(QinvChol, QinvChol.T)
    Q0inv = np.dot(Q0invChol, Q0invChol.T)
    AQinvA = np.dot(np.dot(A.T, Qinv), A)

    AQinvrep = np.kron(np.ones([T-1, 1, 1]), -np.dot(A.T, Qinv))
    AQinvArep = np.kron(np.ones([T-2, 1, 1]), AQinvA + Qinv)
    AQinvArepPlusQ = np.concatenate([np.expand_dims(Q0inv + AQinvA, 0),
                                    AQinvArep, np.expand_dims(Qinv, 0)])

    rm = R.SmoothingLDSTimeSeries(RecognitionParams, Input, 2, 5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = rm.get_summary(Data)
        LambdaChol = rm.LambdaChol.eval(feed_dict={rm.Input: Data})
        LambdaXChol = rm.LambdaXChol.eval(feed_dict={rm.Input: Data})
        Lambda = np.matmul(LambdaChol, np.transpose(LambdaChol, [0, 2, 1]))
        LambdaX = np.matmul(LambdaXChol, np.transpose(LambdaXChol, [0, 2, 1]))
        AA = (Lambda + np.concatenate([np.expand_dims(
          np.zeros([xDim, xDim]), 0), LambdaX]) + AQinvArepPlusQ)
        BB = (np.matmul(LambdaChol[:-1], np.transpose(
          LambdaXChol, [0, 2, 1])) + AQinvrep)
        V, VV, S = sym.compute_sym_blk_tridiag(
          tf.constant(AA.astype(np.float32)),
          tf.constant(BB.astype(np.float32)))

        npt.assert_allclose(output['Vsm'], V.eval(), atol=1e-5, rtol=1e-4)
        npt.assert_allclose(output['VVsm'], VV.eval(), atol=1e-5, rtol=1e-4)


def test_smoothing_past_LDS_time_series_recognition():
    xDim = 2
    yDim = 5
    lag = 5
    Data = np.reshape(np.arange(0, 50), [10, 5])

    Inputt = Data
    for i in range(1, lag + 1):
        lagged = np.concatenate([Inputt[0, :yDim].reshape((1, yDim)),
                                Inputt[:-1, -yDim:]], 0)
        Inputt = np.concatenate([Inputt, lagged], 1)

    Input = tf.placeholder(tf.float32, shape=(None, yDim), name='Input')

    mu_nn = layers.Input((None, yDim * lag + yDim))
    mu_nn_d = (layers.Dense(xDim, activation="linear",
               kernel_initializer=tf.orthogonal_initializer())(mu_nn))
    NN_Mu = models.Model(inputs=mu_nn, outputs=mu_nn_d)

    lambda_nn = layers.Input((None, yDim * lag + yDim))
    lambda_nn_d = (layers.Dense(xDim*xDim, activation="linear",
                   kernel_initializer=tf.orthogonal_initializer())(lambda_nn))
    NN_Lambda = models.Model(inputs=lambda_nn, outputs=lambda_nn_d)

    lambdax_nn = layers.Input((None, yDim * lag + yDim))
    lambdax_nn_d = (layers.Dense(xDim*xDim, activation="linear",
                    kernel_initializer=tf.orthogonal_initializer())(
                    lambdax_nn))
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

    rm = R.SmoothingPastLDSTimeSeries(RecognitionParams, Input, 2, 5, 100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Input1 = rm.Input1.eval(feed_dict={Input: Data})

        npt.assert_allclose(Inputt, Input1, atol=1e-5, rtol=1e-4)


def test_smoothing_time_series_recognition():
    xDim = 2
    yDim = 5
    Input = tf.placeholder(shape=(None, yDim), dtype=tf.float32, name='Input')

    mu_nn = layers.Input((None, yDim))
    mu_nn_d = (layers.Dense(xDim, activation="linear",
               kernel_initializer=tf.random_normal_initializer())(mu_nn))
    NN_Mu = models.Model(inputs=mu_nn, outputs=mu_nn_d)

    lambda_nn = layers.Input((None, yDim))
    lambda_nn_d = (layers.Dense(xDim*xDim, activation="linear",
                   kernel_initializer=tf.random_normal_initializer())
                   (lambda_nn))
    NN_Lambda = models.Model(inputs=lambda_nn, outputs=lambda_nn_d)

    lambdax_nn = layers.Input((None, yDim*2))
    lambdax_nn_d = (layers.Dense(xDim*xDim, activation="linear",
                    kernel_initializer=tf.random_normal_initializer())
                    (lambdax_nn))
    NN_LambdaX = models.Model(inputs=lambdax_nn, outputs=lambdax_nn_d)

    RecognitionParams = ({'NN_Mu': {'network': NN_Mu, 'is_train': None},
                          'NN_Lambda': {'network': NN_Lambda,
                                        'is_train': None},
                          'NN_LambdaX': {'network': NN_LambdaX}})

    Data = np.zeros([10, 5])

    rm = R.SmoothingTimeSeries(RecognitionParams, Input, 2, 5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = rm.get_summary(Data)

        AAChol = rm.AAChol.eval(feed_dict={rm.Input: Data})
        BBChol = rm.BBChol.eval(feed_dict={rm.Input: Data})
        diagsquare = np.matmul(AAChol, np.transpose(AAChol, [0, 2, 1]))
        odsquare = np.matmul(BBChol, np.transpose(BBChol, [0, 2, 1]))
        AA = diagsquare + np.concatenate([np.expand_dims(
          np.zeros([xDim, xDim]), 0), odsquare]) + 1e-6*np.eye(xDim)
        BB = np.matmul(AAChol[:-1], np.transpose(BBChol, [0, 2, 1]))
        V, VV, S = sym.compute_sym_blk_tridiag(
          tf.constant(AA.astype(np.float32)),
          tf.constant(BB.astype(np.float32)))

        npt.assert_allclose(output['Vsm'], V.eval(), atol=1e-5, rtol=1e-4)
        npt.assert_allclose(output['VVsm'], VV.eval(), atol=1e-5, rtol=1e-4)


def test_mean_field_gaussian_recognition():
    xDim = 2
    yDim = 5
    Input = tf.placeholder(shape=(None, yDim), dtype=tf.float32, name='Input')

    mu_nn = layers.Input((None, yDim))
    mu_nn_d = (layers.Dense(xDim, activation="linear",
               kernel_initializer=tf.random_normal_initializer())(mu_nn))
    NN_Mu = models.Model(inputs=mu_nn, outputs=mu_nn_d)

    lambda_nn = layers.Input((None, yDim))
    lambda_nn_d = (layers.Dense(xDim*xDim, activation="linear",
                   kernel_initializer=tf.random_normal_initializer())
                   (lambda_nn))
    NN_Lambda = models.Model(inputs=lambda_nn, outputs=lambda_nn_d)

    RecognitionParams = ({'NN_Mu': {'network': NN_Mu, 'is_train': None},
                          'NN_Lambda': {'network': NN_Lambda,
                                        'is_train': None}})

    Data = np.random.randn(10, 5)

    rm = R.MeanFieldGaussian(RecognitionParams, Input, 2, 5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = rm.get_summary(Data)

        LambdaChol = rm.LambdaChol.eval(feed_dict={rm.Input: Data})
        V = np.matmul(LambdaChol, np.transpose(LambdaChol, [0, 2, 1]))

        npt.assert_allclose(output['Vsm'], V, atol=1e-5, rtol=1e-4)
        assert output['VVsm'].shape == (Data.shape[0]-1, 2, 2)

import numpy as np
import tensorflow as tf
import numpy.testing as npt
import tf_gbds.RecognitionModel as R


def test_recognition_model():
    rm = R.RecognitionModel(None, 5, 10)

    assert rm.xDim == 5
    assert rm.yDim == 10


def test_smoothing_LDS_time_series_recognition():

    xDim = 2
    yDim = 5
    Input = tf.placeholder(tf.float32, shape=(None, yDim), name='Input')
    T = 10

    mu_nn = tf.contrib.keras.layers.Input((None, yDim))
    mu_nn_d = (tf.contrib.keras.layers.Dense(xDim, activation="linear",
               kernel_initializer=tf.orthogonal_initializer())(mu_nn))
    NN_Mu = tf.contrib.keras.models.Model(inputs=mu_nn, outputs=mu_nn_d)

    lambda_nn = tf.contrib.keras.layers.Input((None, yDim))
    lambda_nn_d = (tf.contrib.keras.layers.Dense(xDim*xDim,
                   activation="linear",
                   kernel_initializer=tf.orthogonal_initializer())(lambda_nn))
    NN_Lambda = tf.contrib.keras.models.Model(inputs=lambda_nn,
                                              outputs=lambda_nn_d)

    lambdax_nn = tf.contrib.keras.layers.Input((None, yDim))
    lambdax_nn_d = (tf.contrib.keras.layers.Dense(xDim*xDim,
                    activation="linear",
                    kernel_initializer=tf.orthogonal_initializer())
                    (lambdax_nn))
    NN_LambdaX = tf.contrib.keras.models.Model(inputs=lambdax_nn,
                                               outputs=lambdax_nn_d)

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
        sess.run(tf.initialize_all_variables())
        output = rm.get_summary(Data)
        LambdaChol = rm.LambdaChol.eval(feed_dict={rm.Input: Data})
        LambdaXChol = rm.LambdaXChol.eval(feed_dict={rm.Input: Data})
        Lambda = np.matmul(LambdaChol, np.transpose(LambdaChol, [0, 2, 1]))
        LambdaX = np.matmul(LambdaXChol, np.transpose(LambdaXChol, [0, 2, 1]))
        AA = (Lambda + np.concatenate([np.expand_dims(np.zeros([xDim, xDim]),
                                                      0), LambdaX])
              + AQinvArepPlusQ)
        BB = (np.matmul(LambdaChol[:-1], np.transpose(LambdaXChol, [0, 2, 1]))
              + AQinvrep)
        compute_sym_blk_tridiag = R.sym.compute_sym_blk_tridiag
        V, VV, S = compute_sym_blk_tridiag(tf.constant(AA.astype(np.float32)),
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

    mu_nn = tf.contrib.keras.layers.Input((None, yDim * lag + yDim))
    mu_nn_d = (tf.contrib.keras.layers.Dense(xDim, activation="linear",
               kernel_initializer=tf.orthogonal_initializer())(mu_nn))
    NN_Mu = tf.contrib.keras.models.Model(inputs=mu_nn, outputs=mu_nn_d)

    lambda_nn = tf.contrib.keras.layers.Input((None, yDim * lag + yDim))
    lambda_nn_d = (tf.contrib.keras.layers.Dense(xDim*xDim,
                   activation="linear",
                   kernel_initializer=tf.orthogonal_initializer())(lambda_nn))
    NN_Lambda = tf.contrib.keras.models.Model(inputs=lambda_nn,
                                              outputs=lambda_nn_d)

    lambdax_nn = tf.contrib.keras.layers.Input((None, yDim * lag + yDim))
    lambdax_nn_d = (tf.contrib.keras.layers.Dense(xDim*xDim,
                    activation="linear",
                    kernel_initializer=tf.orthogonal_initializer())
                    (lambdax_nn))
    NN_LambdaX = tf.contrib.keras.models.Model(inputs=lambdax_nn,
                                               outputs=lambdax_nn_d)

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
        sess.run(tf.initialize_all_variables())
        Input1 = rm.Input1.eval(feed_dict={Input: Data})

        npt.assert_allclose(Inputt, Input1, atol=1e-5, rtol=1e-4)

# From generative.lib.GenerativeModel_tf import GenerativeModel, LDS
import numpy as np
import tensorflow as tf
import numpy.testing as npt
from scipy.special import gammaln
import tf_gbds.GenerativeModel as G

np.random.seed(1234)
tf.set_random_seed(1234)


def test_generative_model():
    gm = G.GenerativeModel(None, 5, 10)

    assert isinstance(gm.Xsamp, tf.Tensor)


def test_LDS():
    mm = G.LDS({}, 10, 5)

    assert {'x0', 'Q0Chol', 'A', 'RChol', 'QChol',
            'Xsamp'}.issubset(set(dir(mm)))


def test_LDS_sampling():
    mm = G.LDS({}, 10, 5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        xx = mm.sampleX(100)
        yy = mm.sampleY()
        [x, y] = mm.sampleXY(100)
        assert xx.shape == (100, 10)
        assert yy.shape[-1] == tf.TensorShape([5])
        assert x.shape == (100, 10)
        assert y.shape == (100, 5)


def test_LDS_forward():
    mm = G.LDS({}, 10, 5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [x, y] = mm.sampleXY(100)
        x0 = mm.x0.eval()
        yp = mm.rate.eval(feed_dict={mm.Xsamp: x})

        A = mm.A.eval()
        Rinv = mm.Rinv.eval()
        Lambda = mm.Lambda.eval()
        Lambda0 = mm.Lambda0.eval()
        N = y.shape[0]

        resy = y - yp
        resx = x[1:] - x[:x.shape[0]-1].dot(A.T)
        resx0 = x[0] - x0

        lpdf = -(resy.T.dot(resy) * np.diag(Rinv)).sum()
        lpdf += -(resx.T.dot(resx) * Lambda).sum()
        lpdf += -(resx0.dot(Lambda0).dot(resx0))
        lpdf += N * np.log(Rinv).sum()
        lpdf += (N - 1) * np.linalg.slogdet(Lambda)[1]
        lpdf += np.linalg.slogdet(Lambda0)[1]
        lpdf += -N * (x.shape[1] + y.shape[1]) * np.log(2 * np.pi)
        lpdf *= 0.5

        t_logpdf = mm.evaluateLogDensity(tf.constant(x, tf.float32),
                                         tf.constant(y, tf.float32))

        resX0 = mm.resX0.eval()
        resY = mm.resY.eval()
        resX = mm.resX.eval()
        logpdf = t_logpdf.eval()
        npt.assert_allclose(resX0, resx0)
        npt.assert_allclose(resY, resy)
        npt.assert_allclose(resX, resx)
        assert logpdf < 0
        npt.assert_allclose(logpdf, lpdf, atol=1e-4, rtol=1e-4)


def test_PLDS():
    mm = G.PLDS({'output_nlin': 'exponential'}, 10, 5)

    assert isinstance(mm, G.LDS)
    assert {'x0', 'Q0Chol', 'A', 'RChol', 'QChol',
            'Xsamp'}.issubset(set(dir(mm)))


def test_PLDS_sampling():
    mm = G.PLDS({'output_nlin': 'exponential'}, 10, 5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        xx = mm.sampleX(100)
        [x, y] = mm.sampleXY(100)
        yy = mm.sampleY()
        assert xx.shape == (100, 10)
        assert x.shape == (100, 10)
        assert y.shape == (100, 5)
        assert yy.shape[-1] == tf.TensorShape([5])


def test_PLDS_forward():
    mm = G.PLDS({'output_nlin': 'exponential'}, 10, 5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [x, y] = mm.sampleXY(100)
        x0 = mm.x0.eval()
        yp = mm.rate.eval(feed_dict={mm.Xsamp: x})

        yp_np = np.exp((mm.NN_XtoY(mm.Xsamp)).eval(feed_dict={mm.Xsamp: x}))
        npt.assert_allclose(yp, yp_np, atol=1e-5, rtol=1e-4)

        A = mm.A.eval()
        Lambda = mm.Lambda.eval()
        Lambda0 = mm.Lambda0.eval()
        N = y.shape[0]

        resy = y - yp
        resx = x[1:] - x[:-1].dot(A.T)
        resx0 = x[0] - x0

        lpdf = (-0.5 * resx0.dot(Lambda0).dot(resx0.T) -
                0.5 * (resx * resx.dot(Lambda)).sum() +
                0.5 * np.linalg.slogdet(Lambda)[1] * (N - 1) +
                0.5 * np.linalg.slogdet(Lambda0)[1] -
                0.5 * 10 * np.log(2 * np.pi) * N)
        lpdf += np.sum(y * np.log(yp) - yp - gammaln(y + 1))

        logpdf = mm.evaluateLogDensity(tf.constant(x, tf.float32),
                                       tf.constant(y, tf.float32)).eval()

        resY = mm.resY.eval()
        resX = mm.resX.eval()
        resX0 = mm.resX0.eval()
        npt.assert_allclose(resY, resy)
        npt.assert_allclose(resX, resx)
        npt.assert_allclose(resX0, resx0)
        assert logpdf < 0
        npt.assert_allclose(logpdf, lpdf, atol=1e-4, rtol=1e-4)

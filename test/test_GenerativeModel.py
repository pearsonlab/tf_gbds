# From generative.lib.GenerativeModel_tf import GenerativeModel, LDS
import numpy as np
import tensorflow as tf
import numpy.testing as npt
from scipy.special import gammaln
import tf_gbds.GenerativeModel as G
from tf_gbds.nn_utils import get_network

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


def test_GBDS():
    GenerativeParams = ({'get_states': lambda x: np.hstack([x, x*2]),
                         'NN_postJ_mu': get_network(16, 4, 4, 8, 4),
                         'NN_postJ_sigma': get_network(16, 4, 16, 8, 4),
                         'yCols': np.arange(2),
                         'vel': 2 * np.ones(2)})
    mm = G.GBDS(GenerativeParams, 2, 3)

    mm.init_CGAN(ndims_condition=6, ndims_noise=2, ndims_hidden=8,
                 ndims_data=3, nlayers_G=3, nlayers_D=3, batch_size=16)
    mm.init_GAN(ndims_noise=2, ndims_hidden=8, ndims_data=2, nlayers_G=3,
                nlayers_D=3, batch_size=16)


    assert {'pen_eps', 'pen_sigma', 'pen_g', 'bounds_g', 'PID_params',
            'Kp', 'Ki', 'Kd', 'L', 'sigma', 'eps'}.issubset(set(dir(mm)))
    assert mm.JDim == 4

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        assert mm.Kp.shape.as_list() == [2, 1]
        assert mm.Ki.shape.as_list() == [2, 1]
        assert mm.Kd.shape.as_list() == [2, 1]
        assert mm.L.shape.as_list() == [2, 3]

        sigma = np.log1p(np.exp(-7 * np.ones((1, 2))))
        npt.assert_allclose(mm.sigma.eval(), sigma, atol=1e-5, rtol=1e-4)
        eps = np.log1p(np.exp(-11 * np.ones((1, 2))))
        npt.assert_allclose(mm.eps.eval(), eps, atol=1e-5, rtol=1e-4)

        Y = np.random.rand(16, 3)
        post_g = np.random.rand(17, 2)
        _, next_g_tr, Upred_tr, Ypred_tr = mm.get_preds(Y, training=True,
                                                        post_g=post_g)
        g_stack = tf.constant(np.hstack((post_g[:-1], post_g[1:])), tf.float32)
        postJ_mu = mm.NN_postJ_mu(g_stack).eval()
        postJ_unc_sigma = mm.NN_postJ_sigma(g_stack).eval().reshape((-1, 4, 4))
        postJ_sigma = np.zeros(postJ_unc_sigma.shape)
        for i in range(16):
            postJ_sigma[i] = (np.diag(np.log1p(np.exp(np.diag(
                postJ_unc_sigma[i])))) + postJ_unc_sigma[i] -
                np.triu(postJ_unc_sigma[i]))
        postJ = postJ_mu + np.squeeze(np.matmul(
            postJ_sigma, tf.random_normal([16, 4, 1], seed=1234).eval()), 2)
        npt.assert_allclose(mm.draw_postJ(post_g).eval(), postJ, atol=1e-5, rtol=1e-4)

        J_mu = postJ[:, :2]
        J_lambda = np.log1p(np.exp(postJ[:, 2:]))
        next_g_1 = (post_g[:-1] + J_lambda * J_mu) / (1 + J_lambda)
        npt.assert_allclose(next_g_tr.eval(), next_g_1, atol=1e-5, rtol=1e-4)

        # gen_g = np.random.rand(16, 2)
        # J_gen, next_g_gen, Upred_gen, Ypred_gen = mm.get_preds(Y,
        #                                                        training=False,
        #                                                        post_g=None,
        #                                                        gen_g=gen_g)
        # J_mu = J_gen[:, :2]
        # J_lambda = np.log1p(np.exp(J_gen[:, 2:]))
        # goal = ((gen_g[(-1,)] + J_lambda[(-1,)] * J_mu[(-1,)]) /
        #             (1 + J_lambda[(-1,)]))
        # var = mm.sigma.eval()**2 / (1 + J_lambda[(-1,)])
        # goal += tf.random_normal(goal.shape, seed=1234).eval() * np.sqrt(var)
        # next_g_2 = np.vtack((gen_g[1:], goal))
        # npt.assert_allclose(next_g_gen.eval(), next_g_2, atol=1e-5, rtol=1e-4)

        # post_g0 = tf.constant(np.random.rand(16, 2), tf.float32)
        # npt.assert_allclose(mm.evaluateGANLoss(post_g0, mode='D').eval(),
        #                     mm.GAN_g0.get_discr_cost(
        #                         post_g0, mm.GAN_g0.get_generated_data(
        #                             tf.shape(post_g0)[0].eval(), training=True))
        #                     .eval(),
        #                     atol=1e-5, rtol=1e-4)
        # npt.assert_allclose(mm.evaluateCGANLoss(post_J, states, mode='D').eval(),
        #                     mm.CGAN_J.get_discr_cost(
        #                         post_J, mm.CGAN_J.get_generated_data(
        #                             states, training=True))
        #                     .eval(),
        #                     atol=1e-5, rtol=1e-4)


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

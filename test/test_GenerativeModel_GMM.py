import numpy as np
import tensorflow as tf
import numpy.testing as npt
import tf_gbds.GenerativeModel_GMM as G
from tf_gbds.nn_utils import get_network


def test_GBDS():
    def get_states(x, max_vel=None):
        return np.hstack([x, x*2])

    GenerativeParams = ({'get_states': get_states,
                         'GMM_k': 2,
                         'GMM_net': get_network(16, 4, 10, 8, 4),
                         'yCols': np.arange(2),
                         'vel': 2 * np.ones((2, 1)),
                         'all_vel': 2 * np.ones((20, 2, 1))})
    mm = G.GBDS(GenerativeParams, 2, 3)

    mm.init_GAN(ndims_noise=2, ndims_hidden=8, ndims_data=2, nlayers_G=3,
                nlayers_D=3, batch_size=16)

    assert {'pen_eps', 'pen_sigma', 'pen_g', 'bounds_g', 'PID_params',
            'Kp', 'Ki', 'Kd', 'L', 'sigma', 'eps'}.issubset(set(dir(mm)))
    assert mm.yDim == 2

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
        Y = np.random.rand(16, 2).astype(np.float32)
        states = tf.constant(mm.get_states(Y[:-1]))
        (mu, lmbda, all_w, mu_k, lmbda_k), updates = mm.sample_GMM(states)

        post_g = np.random.rand(17, 2).astype(np.float32)

        # evaluate logdensity
        Y = np.random.rand(17, 2).astype(np.float32)
        U_true_np = np.arctanh((Y[1:, np.arange(2)] - Y[:-1, np.arange(2)]) /
                               (2 * np.ones((1, 2))))
        (all_mu, all_lmbda, all_w, g_pred, Upred,
         Ypred), updates = mm.get_preds(Y[:-1], training=True,
                                        post_g=post_g)
        resU_np = U_true_np - Upred.eval()
        LogDensity_np = -np.sum(resU_np**2 / (2 * mm.eps.eval()**2))
        LogDensity_np -= (0.5 * np.log(2 * np.pi) +
                          np.sum(np.log(mm.eps.eval())))

        w_brdcst = all_w.eval().reshape((-1, mm.GMM_k, 1))
        gmm_res_g = post_g[1:].reshape((-1, 1, mm.yDim)) - g_pred.eval()
        gmm_term = (np.log(w_brdcst + 1e-8) - ((1 + all_lmbda.eval()) / (2 *
                    mm.sigma.eval().reshape((1, 1, -1))**2)) * gmm_res_g**2)
        gmm_term += (0.5 * np.log(1 + all_lmbda.eval()) - 0.5 *
                     np.log(2 * np.pi) -
                     np.log(mm.sigma.eval().reshape((1, 1, -1))))

        npt.assert_allclose(mm.evaluateLogDensity(post_g, Y)[0].eval(),
                            LogDensity_np, atol=1e-5, rtol=1e-4)

import numpy as np
import numpy.testing as npt
from scipy.special import gammaln, psi
import tensorflow as tf
from tensorflow.contrib.keras import layers, models
from tf_gbds.layers import DLGMLayer, PKBiasLayer, PKRowBiasLayer


def test_DLGMLayer():

    xDim = 2
    yDim = 5

    mu_nn = layers.Input((None, yDim))
    mu_nn_d = (layers.Dense(xDim*xDim, activation="linear",
               kernel_initializer=tf.orthogonal_initializer())(mu_nn))
    mu_net = models.Model(inputs=mu_nn, outputs=mu_nn_d)

    u_nn = layers.Input((None, yDim))
    u_nn_d = (layers.Dense(xDim*xDim, activation="linear",
              kernel_initializer=tf.orthogonal_initializer())(u_nn))
    u_net = models.Model(inputs=u_nn, outputs=u_nn_d)

    unc_d_nn = layers.Input((None, yDim))
    unc_d_nn_d = (layers.Dense(xDim*xDim, activation="linear",
                  kernel_initializer=tf.orthogonal_initializer())(unc_d_nn))
    unc_d_net = models.Model(inputs=unc_d_nn, outputs=unc_d_nn_d)

    Data = np.random.randn(10, 5).astype(np.float32)

    rec_nets = ({'mu_net': mu_net, 'u_net': u_net, 'unc_d_net': unc_d_net})

    NN = models.Sequential()
    inputlayer = layers.InputLayer(batch_input_shape=(10, 5))
    NN.add(inputlayer)

    lm = DLGMLayer(NN, 4, rec_nets=rec_nets, k=-1)
    lm.calculate_xi(tf.constant(Data.astype(np.float32)))
    lm.get_ELBO(tf.constant(10.0))

    num_units = 4

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        W = lm.W.eval()
        b = lm.b.eval()
        G = lm.G.eval()
        batch_u = lm.batch_u.eval()
        batch_unc_d = lm.batch_unc_d.eval()
        batch_mu = lm.batch_mu.eval()
        batch_Tr_C_lm = lm.batch_Tr_C.eval()
        batch_ld_C_lm = lm.batch_ld_C.eval()
        batch_R_lm = lm.batch_R.eval()
        get_ELBO_lm = lm.get_ELBO(tf.constant(10.0)).eval()
        activation_lm = lm.call(tf.constant(Data, dtype=tf.float32),
                                use_rec_model=True).eval()

    batch_Tr_C = []
    batch_ld_C = []
    batch_R = []

    batch_u = batch_u.astype(np.float32)
    batch_unc_d = batch_unc_d.astype(np.float32)
    for i in range(batch_u.shape[0]):
        u = batch_u[i]
        unc_d = batch_unc_d[i]
        d = np.log1p(np.exp(np.maximum(unc_d, -15.0)), dtype=np.float32)
        D_inv = np.diag(1.0 / d)
        eta = 1.0 / (u.T.dot(D_inv).dot(u) + 1.0)
        C = D_inv - eta * D_inv.dot(u).dot(u.T).dot(D_inv)
        Tr_C = np.trace(C)
        ld_C = np.log(eta) - np.log(d).sum()  # eq 20 in DLGM
        # coeff = ((1 - T.sqrt(eta)) / (u.T.dot(D_inv).dot(u)))
        # simplified coefficient below is more stable as u -> 0
        # original coefficient from paper is above
        coeff = eta / (1.0 + np.sqrt(eta))
        R = np.sqrt(D_inv) - coeff * D_inv.dot(u).dot(u.T).dot(np.sqrt(D_inv))

        batch_Tr_C.append(Tr_C)
        batch_ld_C.append(ld_C)
        batch_R.append(R)

    batch_Tr_C = np.array(batch_Tr_C)
    batch_ld_C = np.array(batch_ld_C)
    batch_R = np.array(batch_R)

    npt.assert_allclose(batch_Tr_C_lm, batch_Tr_C, atol=1e-3, rtol=1e-4)
    npt.assert_allclose(batch_ld_C_lm, batch_ld_C, atol=1e-3, rtol=1e-4)
    npt.assert_allclose(batch_R_lm, batch_R, atol=1e-3, rtol=1e-4)

    KL_div = (0.5 * (np.sqrt((batch_mu**2).sum(axis=1)).sum() +
              batch_Tr_C.sum() - batch_ld_C.sum() - 10.0))
    weight_reg = ((0.5 / -1) *
                  np.sqrt((W**2).sum()) *
                  np.sqrt((G**2).sum()))

    get_ELBO_np = -(weight_reg + KL_div)

    npt.assert_allclose(get_ELBO_np, get_ELBO_lm, atol=1e-5, rtol=1e-4)

    test_rand = np.random.normal(size=(batch_R.shape[0], num_units))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_mu = lm.batch_mu.eval()
        batch_xi = (batch_mu + np.squeeze(np.matmul(lm.batch_R.eval(),
                    np.expand_dims(test_rand, axis=2))))

        test_batch_xi = (lm.batch_mu + tf.squeeze(tf.matmul(lm.batch_R,
                         tf.expand_dims(tf.constant(test_rand, tf.float32),
                                        -1))))

        activation = np.matmul(np.maximum(Data, 0), W) + b
        xi = batch_xi
        activation += np.matmul(xi, G)

        inputs = tf.constant(Data, dtype=tf.float32)
        activation_lm = tf.matmul(lm.nonlinearity(inputs), lm.W) + lm.b
        activation_lm += tf.matmul(tf.constant(xi, tf.float32), lm.G)
        activation_lm = activation_lm.eval()

        npt.assert_allclose(batch_xi, test_batch_xi.eval(), atol=1e-5,
                            rtol=1e-4)
        npt.assert_allclose(activation_lm, activation, atol=1e-3,
                            rtol=1e-4)


def test_pkbiaslayer():
    batch_size = 16
    num_inputs = 8
    NN = models.Sequential()
    NN.add(layers.InputLayer(batch_input_shape=(batch_size, num_inputs)))
    params = {'k': 1}
    nbatches = 4
    Input = np.random.randn(1, num_inputs)

    l = PKBiasLayer(NN, params)
    assert isinstance(l, layers.Layer)
    assert l.draw_on_every_output

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        npt.assert_array_equal(l.mode.eval(), np.zeros(4))
        s = np.exp(l.log_s.eval())
        npt.assert_allclose(l.s.eval(), s, atol=1e-5, rtol=1e-4)

        biases = l.m.eval() + tf.random_normal(shape=[4, num_inputs],
                                               seed=1234).eval() * s
        npt.assert_allclose(l.biases.eval(), biases, atol=1e-5, rtol=1e-4)

        Input += np.dot(np.zeros(4).reshape((1, -1)), biases)
        npt.assert_allclose(l.call(Input).eval(), Input,
                            atol=1e-5, rtol=1e-4)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        s = np.exp(l.log_s.eval())
        biases = l.m.eval() + tf.random_normal(shape=[4, num_inputs],
                                               seed=1234).eval() * s
        ELBO_tf = (tf.reduce_sum(-tf.abs(tf.constant(biases)) / l.k -
                                 tf.log(tf.constant(2.0) * l.k)))
        ELBO_tf += tf.reduce_sum(tf.log(l.s))
        ELBO_np = (-abs(biases) / params['k'] - np.log(2 * params['k'])).sum()
        ELBO_np += np.log(s).sum()
        npt.assert_allclose((ELBO_tf / nbatches).eval(), ELBO_np / nbatches,
                            atol=1e-5, rtol=1e-4)


def test_pkrowbiaslayer():
    batch_size = 16
    num_inputs = 8
    NN = models.Sequential()
    NN.add(layers.InputLayer(batch_input_shape=(batch_size, num_inputs)))
    params = {'a': 1, 'b': 1}
    nbatches = 4
    Input = np.random.randn(1, num_inputs)

    l = PKRowBiasLayer(NN, params)
    assert isinstance(l, layers.Layer)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        npt.assert_array_equal(l.mode.eval(), np.zeros(4))
        alpha = np.tile((params['a'] + num_inputs / 2), (4, 1))
        npt.assert_array_equal(l.alpha.eval(), alpha)
        beta = np.tile(params['b'], (4, 1))
        npt.assert_array_equal(l.beta.eval(), beta)

        sigma = np.log(np.exp(l.unc_sig.eval()) + 1)
        npt.assert_allclose(l.sigma.eval(), sigma, atol=1e-5, rtol=1e-4)

        gamma = l.mu.eval() + tf.random_normal(shape=[4, num_inputs],
                                               seed=1234).eval() * sigma
        npt.assert_allclose(l.gamma.eval(), gamma, atol=1e-5, rtol=1e-4)

        gamma = l.mu.eval() + tf.random_normal(shape=(4, num_inputs),
                                               seed=1234).eval() * sigma
        Input += np.dot(np.zeros(4).reshape((1, -1)), gamma)
        npt.assert_allclose(l.call(Input).eval(), Input,
                            atol=1e-5, rtol=1e-4)

        beta = params['b'] + 0.5 * (l.mu.eval()**2 + sigma**2).sum(
            axis=1, keepdims=True)  # coord_update
        ELBO = (-0.5 * (l.mu.eval()**2 + sigma**2) * (alpha / beta) +
                0.5 * (psi(alpha) - np.log(beta)) -
                0.5 * np.log(2 * np.pi)).sum()
        ELBO += ((params['a'] - 1) * (psi(alpha) - np.log(beta)) -
                 params['b'] * (alpha / beta) +
                 params['a'] * np.log(params['b']) -
                 gammaln(params['a'])).sum()
        ELBO += (0.5 * np.log(2 * np.pi) + 0.5 + np.log(sigma)).sum()
        ELBO += (alpha - np.log(beta) + gammaln(alpha) +
                 (1 - alpha) * psi(alpha)).sum()
        npt.assert_allclose(l.get_ELBO(nbatches).eval(), ELBO / nbatches,
                            atol=1e-5, rtol=1e-4)

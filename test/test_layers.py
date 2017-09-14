import numpy as np
import numpy.testing as npt
from scipy.special import gammaln, psi
import tensorflow as tf
from tensorflow.contrib.keras import layers, models
from layers import PKBiasLayer, PKRowBiasLayer


def test_pkbiaslayer():
    batch_size = 16
    num_inputs = 8
    NN = models.Sequential()
    NN.add(layers.InputLayer(batch_input_shape=(batch_size, num_inputs)))
    params = {'k': 1}
    nbatches = 4
    Input = np.random.randn(1, num_inputs)

    l = PKBiasLayer(NN, None, params)
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

    l = PKRowBiasLayer(NN, None, params)
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

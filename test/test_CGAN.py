import numpy as np
import numpy.testing as npt
import tensorflow as tf
import tf_gbds.CGAN as C


def test_CGAN():
    nlayers_G = 1
    nlayers_D = 2
    ndims_condition = 3
    ndims_noise = 4
    ndims_hidden = 5
    ndims_data = 5
    batch_size = 10
    training = None

    cg = C.CGAN(nlayers_G, nlayers_D, ndims_condition, ndims_noise,
                ndims_hidden, ndims_data, batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        conditions = np.reshape(np.arange(0, 30), [10, 3])
        if_conditions = np.random.randn(conditions.shape[0],
                                        conditions.shape[1])
        rnoise = np.random.rand(conditions.shape[0], cg.ndims_noise)

        if cg.condition_scale is not None:
            conditions /= cg.condition_scale
        if cg.condition_noise is not None and training:
            conditions += (cg.condition_noise * if_conditions)
        noise = 2 * rnoise - 1
        inp = np.hstack((noise, conditions))
        gen_data = cg.gen_net(tf.constant(inp, tf.float32)).eval()

        if cg.condition_scale is not None:
            conditions /= cg.condition_scale
        if cg.condition_noise is not None and training:
            conditions += (cg.condition_noise *
                           tf.constant(if_conditions, tf.float32))
        noise = 2 * tf.constant(rnoise, tf.float32) - 1
        inp = tf.concat([noise, tf.constant(conditions, tf.float32)], 1)
        gen_data_tf = cg.gen_net(inp).eval()

    npt.assert_allclose(gen_data_tf, gen_data, atol=1e-3, rtol=1e-4)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data = np.reshape(np.arange(0, 50), [10, 5])
        conditions = np.reshape(np.arange(0, 30), [10, 3])
        if cg.condition_scale is not None:
            conditions /= cg.condition_scale
        if cg.condition_noise is not None and training:
            conditions += (cg.condition_noise *
                           np.random.randn(conditions.shape))
        if cg.instance_noise is not None and training:
            data += (cg.instance_noise *
                     np.random.randn((data.shape)))
        inp = np.hstack((data, conditions))
        discr_probs = cg.discr_net(tf.constant(inp, tf.float32)).eval()

        discr_probs_tf = cg.get_discr_vals(tf.constant(data, tf.float32),
                                           tf.constant(conditions,
                                                       tf.float32)).eval()

    npt.assert_allclose(discr_probs_tf, discr_probs, atol=1e-3, rtol=1e-4)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        real_data = np.reshape(np.arange(0, 50), [10, 5])
        fake_data = np.reshape(np.arange(5, 55), [10, 5])
        gen_data = np.reshape(np.arange(10, 60), [10, 5])

        real_discr_out = cg.get_discr_vals(tf.constant(real_data, tf.float32),
                                           tf.constant(conditions, tf.float32),
                                           training=True).eval()
        fake_discr_out = cg.get_discr_vals(tf.constant(fake_data, tf.float32),
                                           tf.constant(conditions, tf.float32),
                                           training=True).eval()
        cost = real_discr_out.mean() - fake_discr_out.mean()

        #  Gradient penalty from "Improved Training of Wasserstein GANs"
        alpha = np.random.rand(cg.batch_size, 1)
        interpolates = (tf.constant(alpha * real_data + (1 - alpha) *
                                    fake_data, tf.float32))
        interp_discr_out = cg.get_discr_vals(interpolates,
                                             tf.constant(conditions,
                                                         tf.float32),
                                             training=True)
        gradients = tf.gradients(tf.reduce_sum(interp_discr_out),
                                 interpolates)[0].eval()
        slopes = np.sqrt((gradients**2).sum(axis=1))  # gradient norms
        gradient_penalty = np.mean((slopes - 1)**2)
        cost -= cg.lmbda * gradient_penalty

        fake_discr_out = cg.get_discr_vals(tf.constant(gen_data, tf.float32),
                                           tf.constant(conditions, tf.float32),
                                           training=True).eval()
        gen_cost = fake_discr_out.mean()

        real_discr_out = cg.get_discr_vals(tf.constant(real_data, tf.float32),
                                           tf.constant(conditions, tf.float32),
                                           training=True)
        fake_discr_out = cg.get_discr_vals(tf.constant(fake_data, tf.float32),
                                           tf.constant(conditions, tf.float32),
                                           training=True)
        cost_tf = (tf.reduce_mean(real_discr_out) -
                   tf.reduce_mean(fake_discr_out))
        alpha = tf.constant(alpha, tf.float32)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interp_discr_out = cg.get_discr_vals(interpolates, conditions,
                                             training=True)
        gradients = tf.gradients(tf.reduce_sum(interp_discr_out),
                                 interpolates)[0]
        slopes = tf.sqrt(tf.reduce_sum(gradients**2, axis=1))  # gradient norms
        gradient_penalty = tf.reduce_mean((slopes - 1)**2)
        cost_tf -= cg.lmbda * gradient_penalty
        cost_tf = cost_tf.eval()
        print(cost_tf)

        gen_cost_tf = cg.get_gen_cost(tf.constant(gen_data, tf.float32),
                                      tf.constant(conditions,
                                                  tf.float32)).eval()

    npt.assert_allclose(cost_tf, cost, atol=1e-3, rtol=1e-4)
    npt.assert_allclose(gen_cost_tf, gen_cost, atol=1e-3, rtol=1e-4)


def test_WGAN():

    nlayers_G = 1
    nlayers_D = 2
    ndims_noise = 4
    ndims_hidden = 5
    ndims_data = 5
    batch_size = 10

    wg = C.WGAN(nlayers_G, nlayers_D, ndims_noise, ndims_hidden, ndims_data,
                batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        real_data = np.reshape(np.arange(0, 50), [10, 5])
        fake_data = np.reshape(np.arange(5, 55), [10, 5])
        gen_data = np.reshape(np.arange(10, 60), [10, 5])

        real_discr_out = wg.get_discr_vals(tf.constant(real_data, tf.float32),
                                           training=True).eval()
        fake_discr_out = wg.get_discr_vals(tf.constant(fake_data, tf.float32),
                                           training=True).eval()
        wcost = real_discr_out.mean() - fake_discr_out.mean()
        alpha = np.random.rand(wg.batch_size, 1)
        interpolates = tf.constant(alpha * real_data + (1 - alpha) * fake_data,
                                   tf.float32)
        interp_discr_out = wg.get_discr_vals(interpolates,
                                             training=True)
        gradients = tf.gradients(tf.reduce_sum(interp_discr_out),
                                 interpolates)[0].eval()
        slopes = np.sqrt((gradients**2).sum(axis=1))  # gradient norms
        gradient_penalty = np.mean((slopes - 1)**2)
        wcost -= wg.lmbda * gradient_penalty

        fake_discr_out = wg.get_discr_vals(tf.constant(gen_data, tf.float32),
                                           training=True).eval()
        gen_wcost = fake_discr_out.mean()

        real_discr_out = wg.get_discr_vals(tf.constant(real_data, tf.float32),
                                           training=True)
        fake_discr_out = wg.get_discr_vals(tf.constant(fake_data, tf.float32),
                                           training=True)
        wcost_tf = (tf.reduce_mean(real_discr_out) -
                    tf.reduce_mean(fake_discr_out))
        alpha = tf.random_uniform((wg.batch_size, 1))
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interp_discr_out = wg.get_discr_vals(interpolates,
                                             training=True)
        gradients = tf.gradients(tf.reduce_sum(interp_discr_out),
                                 interpolates)[0]
        slopes = tf.sqrt(tf.reduce_sum(gradients**2, axis=1))  # gradient norms
        gradient_penalty = tf.reduce_mean((slopes - 1)**2)
        wcost_tf -= wg.lmbda * gradient_penalty
        wcost_tf = wcost_tf.eval()

        gen_wcost_tf = wg.get_gen_cost(tf.constant(gen_data,
                                                   tf.float32)).eval()

    npt.assert_allclose(wcost_tf, wcost, atol=1e-3, rtol=1e-4)
    npt.assert_allclose(gen_wcost_tf, gen_wcost, atol=1e-3, rtol=1e-4)

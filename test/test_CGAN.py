import numpy as np
import numpy.testing as npt
import tensorflow as tf
import numdifftools.nd_algopy as nda
import tf_gbds.CGAN as C


def test_CGAN():
    nlayers_G = 5
    nlayers_D = 10
    ndims_condition = 3
    ndims_noise = 4
    ndims_hidden = 5 
    ndims_data = 5 
    batch_size = 10
    srng = None

    cg = C.CGAN(nlayers_G, nlayers_D, ndims_condition, ndims_noise, ndims_hidden, ndims_data, batch_size, srng)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if cg.condition_scale is not None:
            conditions /= cg.condition_scale
        if cg.condition_noise is not None and training:
            conditions += (cg.condition_noise *
                           np.random.randn(conditions.shape))
        conditions = np.reshape(np.arange(0, 30), [10, 3])
        noise = 2 * np.random.rand(conditions.shape[0],
                                       cg.ndims_noise) - 1
        # noise = tf.random_normal((conditions.shape[0],
        #                           cg.ndims_noise))
        inp = np.hstack((noise, conditions))
        gen_data = cg.gen_net(tf.constant(inp, tf.float32)).eval()

        gen_data_tf = cg.get_generated_data(tf.constant(conditions, tf.float32)).eval()
 
    #npt.assert_allclose(gen_data_tf, gen_data, atol=1e-3, rtol=1e-4)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if cg.condition_scale is not None:
            conditions /= cg.condition_scale
        if cg.condition_noise is not None and training:
            conditions += (cg.condition_noise *
                           np.random.randn(conditions.shape))
        if cg.instance_noise is not None and training:
            data += (cg.instance_noise *
                     np.random.randn((data.shape)))
        data = np.reshape(np.arange(0, 50), [10, 5])
        conditions = np.reshape(np.arange(0, 30), [10, 3])
        inp = np.hstack((data, conditions))
        discr_probs = cg.discr_net(tf.constant(inp, tf.float32)).eval()

        discr_probs_tf = cg.get_discr_vals(tf.constant(data, tf.float32), tf.constant(conditions, tf.float32)).eval()

    npt.assert_allclose(discr_probs_tf, discr_probs, atol=1e-3, rtol=1e-4) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        real_data = np.reshape(np.arange(0, 50),[10, 5])
        fake_data = np.reshape(np.arange(5, 55),[10, 5])
        gen_data = np.reshape(np.arange(10, 60),[10, 5])
        real_discr_out = cg.get_discr_vals(tf.constant(real_data, tf.float32), tf.constant(conditions, tf.float32),
                                             training=True).eval()
        fake_discr_out = cg.get_discr_vals(tf.constant(fake_data, tf.float32), tf.constant(conditions, tf.float32),
                                             training=True).eval()
        cost = real_discr_out.mean() - fake_discr_out.mean()

        #  Gradient penalty from "Improved Training of Wasserstein GANs"
        alpha = np.random.rand(cg.batch_size, 1)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        print(interpolates)
        interp_discr_out = cg.get_discr_vals(tf.constant(interpolates, tf.float32), tf.constant(conditions, tf.float32),
                                               training=True)
        print(interp_discr_out)

        gradients = tf.gradients(tf.reduce_sum(interp_discr_out), (tf.constant(interpolates, tf.float32)))
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            grad_value = sess.run(grad)
            print(grad_value)
        

        slopes = np.sqrt((gradients**2).sum(axis=1))  # gradient norms
        gradient_penalty = np.mean((slopes - 1)**2)
        cost -= cg.lmbda * gradient_penalty

        
        fake_discr_out = cg.get_discr_vals(tf.constant(gen_data, tf.float32), tf.constant(conditions, tf.float32),
                                             training=True).eval()
        gen_cost = fake_discr_out.mean()


        cost_tf = cg.get_discr_cost(tf.constant(real_data,tf.float32), tf.constant(fake_data,tf.float32), tf.constant(conditions, tf.float32)).eval()
        gen_cost_tf = cg.get_gen_cost(tf.constant(gen_data, tf.float32), tf.constant(conditions, tf.float32)).eval()

    npt.assert_allclose(cost_tf, cost, atol=1e-3, rtol=1e-4)
    npt.assert_allclose(gen_cost_tf, gen_cost, atol=1e-3, rtol=1e-4)


def test_WGAN():

    nlayers_G = 1
    nlayers_D = 2
    ndims_condition = 3
    ndims_noise = 4
    ndims_hidden = 5 
    ndims_data = 5 
    batch_size = 10
    srng = None

    wg = C.WGAN(nlayers_G, nlayers_D, ndims_noise, ndims_hidden, ndims_data, batch_size, srng)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        real_data = np.reshape(np.arange(0, 50),[10, 5])
        fake_data = np.reshape(np.arange(5, 55),[10, 5])
        gen_data = np.reshape(np.arange(10, 60),[10, 5])

        real_discr_out = wg.get_discr_vals(tf.constant(real_data, tf.float32),
                                             training=True).eval()
        fake_discr_out = wg.get_discr_vals(tf.constant(fake_data, tf.float32),
                                             training=True).eval()
        cost = real_discr_out.mean() - fake_discr_out.mean()

        #  Gradient penalty from "Improved Training of Wasserstein GANs"
        alpha = np.random.rand(wg.batch_size, 1)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interp_discr_out = wg.get_discr_vals(tf.constant(interpolates, tf.float32),
                                               training=True).eval()
        gradients = np.gradient(interp_discr_out.sum(), interpolates)
        slopes = np.sqrt((gradients**2).sum(axis=1))  # gradient norms
        gradient_penalty = np.mean((slopes - 1)**2)
        wcost -= wg.lmbda * gradient_penalty

        fake_discr_out = wg.get_discr_vals(tf.constant(gen_data, tf.float32),
                                             training=True).eval()        
        gen_wcost = fake_discr_out.mean()

        wcost_tf = wg.get_discr_cost(tf.constant(real_data,tf.float32), tf.constant(fake_data,tf.float32))
        gen_wcost_tf = wg.get_gen_cost(tf.constant(gen_data,tf.float32))


    npt.assert_allclose(wcost_tf, wcost, atol=1e-3, rtol=1e-4)
    npt.assert_allclose(gen_wcost_tf, gen_wcost, atol=1e-3, rtol=1e-4)







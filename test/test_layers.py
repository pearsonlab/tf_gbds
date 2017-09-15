import numpy as np
import tensorflow as tf
import numpy.testing as npt
import tf_gbds.layers as L

np.random.seed(1234)
tf.set_random_seed(1234)

def test_DLGMLayer():
    
    xDim = 2
    yDim = 5
    
    
    mu_nn = tf.contrib.keras.layers.Input((None, yDim))
    mu_nn_d = (tf.contrib.keras.layers.Dense(xDim*xDim, activation="linear",
               kernel_initializer=tf.orthogonal_initializer())(mu_nn))
    mu_net = tf.contrib.keras.models.Model(inputs=mu_nn, outputs=mu_nn_d)

    u_nn = tf.contrib.keras.layers.Input((None, yDim))
    u_nn_d = (tf.contrib.keras.layers.Dense(xDim*xDim,
                   activation="linear",
                   kernel_initializer=tf.orthogonal_initializer())(u_nn))
    u_net = tf.contrib.keras.models.Model(inputs=u_nn,
                                              outputs=u_nn_d)

    unc_d_nn = tf.contrib.keras.layers.Input((None, yDim))
    unc_d_nn_d = (tf.contrib.keras.layers.Dense(xDim*xDim,
                    activation="linear",
                    kernel_initializer=tf.orthogonal_initializer())
                    (unc_d_nn))
    unc_d_net = tf.contrib.keras.models.Model(inputs=unc_d_nn,
                                               outputs=unc_d_nn_d)

    Data = np.reshape(np.arange(0, 50), [10, 5])

    A = .5*np.diag(np.ones(xDim))
    QinvChol = np.eye(xDim)
    Q0invChol = np.eye(xDim)

    rec_nets = ({'mu_net': mu_net,
                          'u_net': u_net,
                          'unc_d_net': unc_d_net
                         })


    NN = tf.contrib.keras.models.Sequential()
    inputlayer = tf.contrib.keras.layers.InputLayer(batch_input_shape=(10, 5))
    NN.add(inputlayer)

    lm = L.DLGMLayer(NN, 4, srng=None, rec_nets=rec_nets, k=-1)
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
        activation_lm = lm.call(tf.constant(Data, dtype=tf.float32),use_rec_model=True).eval()
    
    
    batch_Tr_C = []
    batch_ld_C = []
    batch_R = []
    
    batch_u = batch_u.astype(np.float32)
    batch_unc_d = batch_unc_d.astype(np.float32)
    for i in range(batch_u.shape[0]):
        u = batch_u[i]
        unc_d = batch_unc_d[i]
        d = np.log(1 + np.exp(np.maximum(unc_d, -15.0)))
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

    npt.assert_allclose(batch_Tr_C_lm, batch_Tr_C , atol=1e-3, rtol=1e-4)
    npt.assert_allclose(batch_ld_C_lm, batch_ld_C , atol=1e-3, rtol=1e-4)
    npt.assert_allclose(batch_R_lm, batch_R , atol=1e-3, rtol=1e-4)


    KL_div = 0.5 * (np.sqrt((batch_mu**2).sum(axis=1)).sum() +
                      batch_Tr_C.sum() - batch_ld_C.sum() -
                      10.0)
    weight_reg = ((0.5 / -1) *
                  np.sqrt((W**2).sum()) *
                  np.sqrt((G**2).sum()))
    
    get_ELBO_np = -(weight_reg + KL_div)

    npt.assert_allclose(get_ELBO_np, get_ELBO_lm , atol=1e-5, rtol=1e-4)
 
    
    test_rand = np.random.normal(size=(batch_R.shape[0],
                                                          num_units))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_mu = lm.batch_mu.eval()
        test_rand = np.reshape(np.arange(0, 40, dtype=np.float32), [10, 4])
        batch_xi = (batch_mu +
                     np.squeeze(np.matmul(lm.batch_R.eval(), np.expand_dims(test_rand,axis = 2))))
       
        
        test_batch_xi = (lm.batch_mu + tf.squeeze(tf.matmul(lm.batch_R,
                                       tf.expand_dims(tf.constant(test_rand),-1))))

        activation = np.matmul(Data.astype(np.float32), W) + b
        xi = batch_xi
        activation += np.matmul(xi, G)
        
        inputs = tf.constant(Data, dtype=tf.float32)
        activation_lm = tf.matmul(lm.nonlinearity(inputs), lm.W) + lm.b
        xi_lm = test_batch_xi

        activation_lm += tf.matmul(xi, lm.G)


        npt.assert_allclose(batch_xi, test_batch_xi.eval() , atol=1e-5, rtol=1e-4)
        npt.assert_allclose(activation_lm.eval(), activation , atol=1e-3, rtol=1e-4)
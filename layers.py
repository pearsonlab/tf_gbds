"""
Base class for a generative model and linear dynamical system implementation.
Based on Evan Archer's code here: https://github.com/earcher/vilds/blob/master/code/layers.py
"""
import tensorflow as tf
import numpy as np

#def default_param_init(shape, dtype = None):
#    par_init = tf.contrib.keras.initializers.RandomNormal(0, 0.01)
#    return par_init(shape, dtype = dtype)

class DLGMLayer(tf.contrib.keras.layers.Layer):
    """
    This layer is inspired by the paper "Stochastic Backpropagation and
    Approximate Inference in Deep Generative Models"

    incoming (Lasagne Layer): preceding layer in DLGM
    num_units (int): number of output units in this layer
    srng (theano RandomState): random number generator
    rec_nets (dictionary of lasagne NNs): Neural networks that
        paramaterize the recognition model
    J (theano symbolic matrix): Input to rec model
    k (float): regularization term on generative weights
    """
    def __init__(self, incoming, num_units, srng, rec_nets, k,
                 output_layer=False, extra_noise=0.01,
                 param_init=tf.random_normal_initializer(0, 0.01, seed=1234),
                 nonlinearity=tf.nn.relu,
                 **kwargs):
        super(DLGMLayer, self).__init__(**kwargs)
        
        num_inputs = incoming.output_shape[1]
        # num_inputs = self.input.get_shape[1]
        self.srng = srng
        self.num_units = num_units
        self.output_layer = output_layer
        self.extra_noise = extra_noise

        # Initialize generative/decoding Parameters
        self.W = self.add_variable(name='W', shape=(num_inputs, num_units), initializer=param_init)
        self.b = self.add_variable(name='b', shape=(num_units,), initializer=param_init)
        self.unc_G = self.add_variable(name='unc_G', shape=(num_units, num_units), initializer=param_init)
        self.G = tf.diag(tf.nn.softplus(tf.diag_part(self.unc_G))) + tf.matrix_band_part(self.unc_G, -1, 0)
        self.nonlinearity = nonlinearity

        # regularization term
        self.k = k

        # Load recognition/encoding Parameters
        self.mu_net = rec_nets['mu_net']
        self.u_net = rec_nets['u_net']
        self.unc_d_net = rec_nets['unc_d_net']
        
        # add parameters to layer class
        rec_params = (self.mu_net.variables +
                      self.u_net.variables +
                      self.unc_d_net.variables)
        
        #i = 0
        #self.add_variable(name="param", shape=None, initializer=rec_params[i])
        
        #i = 0
        #for param in rec_params:
        #    self.add_variable(name="param"+str(i), shape=None, initializer=param)
        #    i +=1
   
        #def add_variable_loop_fn(param):
        #    self.add_variable(name=param.name, shape=None, initializer=param)
        #    return None
        # tf.map_fn(add_variable_loop_fn, rec_params, parallel_iterations=1)

    def build(self, incoming, postJ):
    
        rec_params = (self.mu_net.variables +
                      self.u_net.variables +
                      self.unc_d_net.variables)
        
        i = 0
        for param in rec_params:
            self.add_variable(name="param"+str(i), shape=None, initializer=param)
            i +=1

        super(DLGMLayer, self).build(incoming)



    def calculate_xi(self, postJ):
        """
        Calculate xi based on sampled J from posterior
        """
        # get output of rec model
        self.batch_mu = self.mu_net(postJ)
        self.batch_u = self.u_net(postJ)
        self.batch_unc_d = self.unc_d_net(postJ)

        # add extra dim to batch_u, so it gets treated as column vectors when
        # iterated over
        #self.batch_u = tf.reshape(self.batch_u,
        #    [tf.shape(self.batch_u)[0], tf.shape(self.batch_u)[1], 1])
        
        self.batch_u = tf.expand_dims(self.batch_u,-1)
        
        def get_cov(acc, inputs):
            # convert output of rec model to rank-1 covariance matrix

            # use softplus to get positive constrained d, minimum of -15
            # since softplus will turn low numbers into 0, which become NaNs
            # when inverted
            u, unc_d = inputs
            #d = tf.nn.softplus(tf.maximum(unc_d, -15.0))
            d = tf.log(1+tf.exp(tf.maximum(unc_d, -15.0)))
            D_inv = tf.diag(1.0 / d)
            eta = 1.0 / (tf.matmul(tf.matmul(tf.transpose(u), D_inv), u) + 1.0)
            C = D_inv - eta*tf.matmul(tf.matmul(tf.matmul(D_inv, u), tf.transpose(u)), D_inv)
            Tr_C = tf.trace(C)
            ld_C = tf.log(eta) - tf.reduce_sum(tf.log(d)) # eq 20 in DLGM
            # coeff = ((1 - T.sqrt(eta)) / (u.T.dot(D_inv).dot(u)))
            # simplified coefficient below is more stable as u -> 0
            # original coefficient from paper is above
            coeff = eta / (1.0 + tf.sqrt(eta))
            R = tf.sqrt(D_inv) - coeff * tf.matmul(tf.matmul(tf.matmul(D_inv, u), tf.transpose(u)), tf.sqrt(D_inv))
            return Tr_C, ld_C, R
              
        (self.batch_Tr_C, self.batch_ld_C, self.batch_R)= tf.scan(
            get_cov, [self.batch_u, self.batch_unc_d], initializer=(0.0,tf.zeros([1,1]),tf.diag(self.batch_unc_d[0])))
        
        self.batch_xi = (self.batch_mu +
                         tf.squeeze(tf.matmul(self.batch_R,
                                       tf.expand_dims(tf.random_normal(
                                            [tf.shape(self.batch_R)[0],
                                             self.num_units]),-1))))

    def call(self, inputs, add_noise=False, use_rec_model=False):
        activation = tf.matmul(self.nonlinearity(inputs), self.W) + self.b

        if use_rec_model:
            # use sample from rec model
            xi = self.batch_xi
            if add_noise:  # additional noise
                xi += self.extra_noise * tf.random_normal(tf.shape(self.batch_xi))
        else:
            # pure random input
            xi = tf.random_normal((tf.shape(inputs)[0], self.num_units))
        # we want the mean when training, so don't add noise to
        # output of last layer when training.
        if not self.output_layer:
            activation += tf.matmul(xi, self.G)
        elif not add_noise:
            activation += tf.matmul(xi, self.G)

        return activation

    def get_ELBO(self, length):
        """
        Get ELBO for this layer

        length (theano symbolic int): length of current batch
        """
        #  KL divergence between posterior and N(0,1) prior

        
        KL_div = 0.5 * (tf.reduce_sum(tf.sqrt(tf.reduce_sum(self.batch_mu**2, axis=1))) +
                        tf.reduce_sum(self.batch_Tr_C) - tf.reduce_sum(self.batch_ld_C) -
                        length)
        weight_reg = ((0.5 / self.k) *
                      tf.reduce_sum(tf.sqrt(self.W**2)) *
                      tf.reduce_sum(tf.sqrt(self.G**2)))
        return -(weight_reg + KL_div)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_units)


class PKBiasLayer(tf.contrib.keras.layers.Layer):
    """
    This layer draws different biases (depending on the mode)
    from a normal distribution, then adds them to the input

    Default modes are as follows:
    0: normal, no biases added
    1: saline and DLPFC, bias 0 is added
    2: saline and DMPFC, bias 1 is added
    3: muscimol and DLPFC, biases 0 and 2 are added
    4: muscimol and DMPFC, biases 1 and 3 are added
    """
    def __init__(self, incoming, srng, params, param_init=tf.random_normal_initializer(mean=0,stddev=0.01),
                 num_biases=4, **kwargs):
        super(PKBiasLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.mode = tf.zeros(num_biases)
        self.srng = srng
        self.k = np.cast[np.float32](params['k'])

        self.m = self.add_variable(name='m', shape=[num_biases, num_inputs], initializer=param_init)
        self.log_s = self.add_variable(name='log_s', shape=[num_biases, num_inputs], initializer=param_init)
        # standard deviation will always be positive but optimization over
        # log_s can be unconstrained
        self.s = tf.exp(self.log_s)
        self.draw_biases()
        self.draw_on_every_output = True

    def set_mode(self, mode):
        self.mode = mode

    def draw_biases(self):
        self.biases = self.m + tf.random_normal(tf.shape(self.s)) * self.s

    def get_ELBO(self, nbatches):
        """
        Return the contribution to the ELBO for these biases

        Normalized by nbatches (number of batches in dataset)
        """
        ELBO = (-tf.abs_(self.biases) / self.k - tf.reduce_sum(tf.log(2 * self.k)))
        ELBO += tf.reduce_sum(tf.log(self.s))
        return ELBO / nbatches

    def get_output_for(self, input, **kwargs):
        if self.draw_on_every_output:
            self.draw_biases()
        act_biases = tf.matmul(tf.reshape(self.mode.astype(tf.float32),[1, -1]), self.biases)

        return input + act_biases

import tensorflow as tf
from tf_gbds.nn_utils import get_network


class CGAN(object):
    """
    Conditional Generative Adversarial Network
    Mirza, Mehdi, Osindero, Simon, Conditional generative adversarial nets,
    Arxiv:1411.1784v1, 2014

    Uses Improved Wasserstein GAN formulation
    Gulrajani, Ishaan, et al. "Improved Training of Wasserstein GANs."
    arXiv preprint arXiv:1704.00028 (2017).
    """
    def __init__(self, nlayers_G, nlayers_D, ndims_condition, ndims_noise,
                 ndims_hidden, ndims_data, batch_size,
                 lmbda=10.0,
                 nonlinearity=tf.contrib.keras.activations.relu,
                 init_std_G=1.0, init_std_D=0.005, condition_noise=None,
                 condition_scale=None, instance_noise=None):
        # Neural network (G) that generates data to match the real data
        self.gen_net = get_network(batch_size,
                                   ndims_condition + ndims_noise, ndims_data,
                                   ndims_hidden, nlayers_G,
                                   init_std=init_std_G,
                                   hidden_nonlin=nonlinearity,
                                   batchnorm=True)
        # Neural network (D) that discriminates between real and generated data
        self.discr_net = get_network(batch_size,
                                     ndims_condition + ndims_data, 1,
                                     ndims_hidden, nlayers_D,
                                     init_std=init_std_D,
                                     hidden_nonlin=nonlinearity,
                                     batchnorm=False)
        # lambda hyperparam (scale of gradient penalty)
        self.lmbda = lmbda
        # size of minibatches (number of rows)
        self.batch_size = batch_size
        # symbolic random number generator
        # self.srng = None
        # number of dimensions of conditional input
        self.ndims_condition = ndims_condition
        # number of dimensions of noise input
        self.ndims_noise = ndims_noise
        # number of hidden units
        self.ndims_hidden = ndims_hidden
        # number of dimensions in the data
        self.ndims_data = ndims_data
        # scale of added noise to conditions as regularization during training
        self.condition_noise = condition_noise
        # scale of each condition dimension, used for normalization
        self.condition_scale = condition_scale
        # scale of added noise to data input into discriminator
        # http://www.inference.vc/instance-noise-a-trick-for-stabilising-
        # gan-training/
        self.instance_noise = instance_noise

    def get_generated_data(self, conditions, training=False):
        """
        Return generated sample from G given conditions.
        """
        if self.condition_scale is not None:
            conditions /= self.condition_scale
        if self.condition_noise is not None and training:
            conditions += (self.condition_noise *
                           tf.random_normal(tf.shape(conditions)))
        noise = 2 * tf.random_uniform([tf.shape(conditions)[0],
                                       self.ndims_noise]) - 1
        # noise = tf.random_normal((conditions.shape[0],
        #                           self.ndims_noise))
        inp = tf.concat([noise, conditions], 1)
        gen_data = self.gen_net(inp)

        return gen_data

    def get_discr_vals(self, data, conditions, training=False):
        """
        Return probabilities of being real data from discriminator network,
        given conditions
        """
        if self.condition_scale is not None:
            conditions /= self.condition_scale
        if self.condition_noise is not None and training:
            conditions += (self.condition_noise *
                           tf.random_normal(tf.shape(conditions)))
        if self.instance_noise is not None and training:
            data += (self.instance_noise *
                     tf.random_normal((tf.shape(data))))
        inp = tf.concat([data, conditions], 1)
        discr_probs = self.discr_net(inp)

        return discr_probs

    def get_gen_params(self):
        return self.gen_net.variables

    def get_discr_params(self):
        return self.discr_net.variables

    def get_discr_cost(self, real_data, fake_data, conditions):
        real_discr_out = self.get_discr_vals(real_data, conditions,
                                             training=True)
        fake_discr_out = self.get_discr_vals(fake_data, conditions,
                                             training=True)
        cost = tf.reduce_mean(real_discr_out) - tf.reduce_mean(fake_discr_out)

        #  Gradient penalty from "Improved Training of Wasserstein GANs"
        alpha = tf.random_uniform((self.batch_size, 1))
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interp_discr_out = self.get_discr_vals(interpolates, conditions,
                                               training=True)
        gradients = tf.gradients(tf.reduce_sum(interp_discr_out), interpolates)
        slopes = tf.sqrt(tf.reduce_sum(gradients**2, axis=1))  # gradient norms
        gradient_penalty = tf.reduce_mean((slopes - 1)**2)
        cost -= self.lmbda * gradient_penalty
        return cost

    def get_gen_cost(self, gen_data, conditions):
        fake_discr_out = self.get_discr_vals(gen_data, conditions,
                                             training=True)
        cost = tf.reduce_mean(fake_discr_out)
        return cost


class WGAN(object):
    """
    Wasserstein Generative Adversarial Network

    Uses Improved Wasserstein GAN formulation
    Gulrajani, Ishaan, et al. "Improved Training of Wasserstein GANs."
    arXiv preprint arXiv:1704.00028 (2017).
    """
    def __init__(self, nlayers_G, nlayers_D, ndims_noise,
                 ndims_hidden, ndims_data, batch_size,
                 lmbda=10.0,
                 nonlinearity=tf.contrib.keras.activations.relu,
                 init_std_G=1.0, init_std_D=0.005, instance_noise=None):
        # Neural network (G) that generates data to match the real data
        self.gen_net = get_network(batch_size, ndims_noise, ndims_data,
                                   ndims_hidden, nlayers_G,
                                   init_std=init_std_G,
                                   hidden_nonlin=nonlinearity,
                                   batchnorm=True)
        # Neural network (D) that discriminates between real and generated
        # data
        self.discr_net = get_network(batch_size, ndims_data, 1, ndims_hidden,
                                     nlayers_D, init_std=init_std_D,
                                     hidden_nonlin=nonlinearity,
                                     batchnorm=False)
        # lambda hyperparam (scale of gradient penalty)
        self.lmbda = lmbda
        # size of minibatches (number of rows)
        self.batch_size = batch_size
        # symbolic random number generator
        # self.srng = None
        # number of dimensions of noise input
        self.ndims_noise = ndims_noise
        # number of hidden units
        self.ndims_hidden = ndims_hidden
        # number of dimensions in the data
        self.ndims_data = ndims_data
        # scale of added noise to data input into discriminator
        # http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-
        # training/
        self.instance_noise = instance_noise

    def get_generated_data(self, N, training=False):
        """
        Return N generated samples from G.
        """
        noise = 2 * tf.random_uniform((N, self.ndims_noise)) - 1
        # noise = tf.random_normal((N, self.ndims_noise))
        gen_data = self.gen_net(noise)
        return gen_data

    def get_discr_vals(self, data, training=False):
        """
        Return probabilities of being real data from discriminator network
        """
        if self.instance_noise is not None and training:
            data += (self.instance_noise *
                     tf.random_normal((data.shape)))
        discr_probs = self.discr_net(data)

        return discr_probs

    def get_gen_params(self):
        return self.gen_net.variables

    def get_discr_params(self):
        return self.discr_net.variables

    def get_discr_cost(self, real_data, fake_data):
        real_discr_out = self.get_discr_vals(real_data,
                                             training=True)
        fake_discr_out = self.get_discr_vals(fake_data,
                                             training=True)
        cost = tf.reduce_mean(real_discr_out) - tf.reduce_mean(fake_discr_out)

        #  Gradient penalty from "Improved Training of Wasserstein GANs"
        alpha = tf.random_uniform((self.batch_size, 1))
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interp_discr_out = self.get_discr_vals(interpolates, training=True)
        gradients = tf.gradients(tf.reduce_sum(interp_discr_out),
                                 interpolates)
        slopes = tf.sqrt(tf.reduce_sum(gradients**2, axis=1))  # gradient norms
        gradient_penalty = tf.reduce_mean((slopes - 1)**2)
        cost -= self.lmbda * gradient_penalty
        return cost

    def get_gen_cost(self, gen_data):
        fake_discr_out = self.get_discr_vals(gen_data, training=True)
        cost = tf.reduce_mean(fake_discr_out)
        return cost

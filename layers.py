"""
The MIT License (MIT)
Copyright (c) 2017 Shariq Iqbal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import backend, layers


class PKBiasLayer(layers.Layer):
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
    def __init__(self, incoming, srng, params,
                 param_init=tf.random_normal_initializer(stddev=0.01),
                 num_biases=4, **kwargs):
        super(PKBiasLayer, self).__init__(**kwargs)

        num_inputs = incoming.output_shape[1]
        self.mode = tf.zeros(num_biases)
        self.srng = srng
        self.k = np.cast[backend.floatx()](params['k'])

        self.m = self.add_variable(name='m', shape=[num_biases, num_inputs],
                                   initializer=param_init)
        self.log_s = self.add_variable(name='log_s',
                                       shape=[num_biases, num_inputs],
                                       initializer=param_init)
        # standard deviation will always be positive but optimization over
        # log_s can be unconstrained
        self.s = tf.exp(self.log_s)

        self.draw_biases()
        self.draw_on_every_output = True

    def build(self, incoming):
        if self.draw_on_every_output:
            self.draw_biases()
        super(PKBiasLayer, self).build(incoming)

    def draw_biases(self):
        self.biases = self.m + tf.random_normal(shape=self.s.shape,
                                                seed=1234) * self.s

    def call(self, inputs):
        act_biases = tf.matmul(tf.reshape(tf.cast(
            self.mode, backend.floatx()), [1, -1]), self.biases)
        return inputs + act_biases

    def set_mode(self, mode):
        self.mode = mode

    def get_ELBO(self, nbatches):
        """
        Return the contribution to the ELBO for these biases

        Normalized by nbatches (number of batches in dataset)
        """
        ELBO = (tf.reduce_sum(-tf.abs(self.biases) / self.k -
                              tf.log(tf.constant(2.0) * self.k)))
        ELBO += tf.reduce_sum(tf.log(self.s))
        return ELBO / nbatches


class PKRowBiasLayer(layers.Layer):
    """
    This layer draws different biases (depending on the mode)
    from a normal distribution, then adds them to the input.
    This layer has sparsity at the row level, instead of the individual
    sparsity of the PKBiasLayer.

    Default modes are as follows:
    0: normal, no biases added
    1: saline and DLPFC, bias 0 is added
    2: saline and DMPFC, bias 1 is added
    3: muscimol and DLPFC, biases 0 and 2 are added
    4: muscimol and DMPFC, biases 1 and 3 are added
    """
    def __init__(self, incoming, srng, params,
                 param_init=tf.random_normal_initializer(stddev=0.01),
                 num_biases=4, **kwargs):
        super(PKRowBiasLayer, self).__init__(**kwargs)
        num_inputs = incoming.output_shape[1]
        self.mode = tf.zeros(num_biases)
        self.srng = srng
        # parameters on prior
        self.a = np.cast[backend.floatx()](params['a'])  # shape
        self.b = np.cast[backend.floatx()](params['b'])  # rate

        # learnable posterior parameters
        # normal dist over biases
        self.mu = self.add_variable(name='mu', shape=[num_biases, num_inputs],
                                    initializer=param_init)

        self.unc_sig = self.add_variable(name='unc_sig',
                                         shape=[num_biases, num_inputs],
                                         initializer=param_init)

        # gamma over rows
        self.alpha = tf.Variable(initial_value=self.a * np.ones(
            (num_biases, 1)), name='alpha', dtype=tf.float32)
        self.beta = tf.Variable(initial_value=self.b * np.ones(
            (num_biases, 1)), name='beta', dtype=tf.float32)

        # update for alpha
        self.alpha += (num_inputs / 2.0)

        # standard deviation will always be positive but optimization over
        # unc_sig can be unconstrained
        self.sigma = tf.nn.softplus(self.unc_sig)

        self.draw_biases()
        self.draw_on_every_output = True

    def build(self, incoming):
        if self.draw_on_every_output:
            self.draw_biases()
        super(PKRowBiasLayer, self).build(incoming)

    def draw_biases(self):
        self.gamma = self.mu + tf.random_normal(
            shape=self.sigma.shape, seed=1234) * self.sigma

    def call(self, input):
        act_biases = tf.matmul(tf.reshape(tf.cast(
            self.mode, backend.floatx()), [1, -1]), self.gamma)
        return input + act_biases

    def set_mode(self, mode):
        self.mode = mode

    def coord_update(self):
        self.beta = self.b + 0.5 * tf.reduce_sum(self.mu**2 + self.sigma**2,
                                                 axis=1,
                                                 keep_dims=True)

    def get_ELBO(self, nbatches):
        """
        Return the contribution to the ELBO for these biases

        Normalized by nbatches (number of batches in dataset)
        """
        self.coord_update()
        # Log Density
        ELBO = (tf.reduce_sum(-0.5 * (self.mu**2 + self.sigma**2) *
                (self.alpha / self.beta) + 0.5 * (tf.digamma(self.alpha) -
                tf.log(self.beta)) - 0.5 * tf.log(2 * np.pi)))
        ELBO += (tf.reduce_sum((self.a - 1) * (tf.digamma(self.alpha) -
                 tf.log(self.beta)) - self.b * (self.alpha / self.beta) +
                 self.a * tf.log(self.b) - tf.lgamma(self.a)))
        # entropy
        ELBO += (tf.reduce_sum(0.5 * tf.log(2 * np.pi) + 0.5 +
                 tf.log(self.sigma)))
        ELBO += (tf.reduce_sum(self.alpha - tf.log(self.beta) +
                 tf.lgamma(self.alpha) + (1 - self.alpha) *
                 tf.digamma(self.alpha)))
        return ELBO / nbatches

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

rewrite in tensorflow
"""
import tensorflow as tf
# from keras import backend as K
from tensorflow.contrib.keras.models import Sequential
from tensorflow.contrib.keras.layers import (Activation, BatchNormalization,
                                             Conv1D, Dense, InputLayer,
                                             Lambda, ZeroPadding1D)
from layers import PKBiasLayer, PKRowBiasLayer


def get_network(batch_size, input_dim, output_dim, hidden_dim, num_layers,
                PKLparams=None, srng=None, batchnorm=False, is_shooter=False,
                row_sparse=False, add_pklayers=False, filt_size=None,
                hidden_nonlin="relu", output_nonlin="linear",
                init_std=1.0):
    """
    Returns a NN with the specified parameters.
    Also returns a list of PKBias layers
    """
    PKbias_layers = []
    NN = Sequential()
    # K.set_learning_phase(0)
    NN.add(InputLayer(batch_input_shape=(batch_size, input_dim), name="Input"))
    if filt_size is not None:  # first layer convolution
        # expand dims for convolution
        NN.add(Lambda(lambda x: tf.expand_dims(x, 0), name="ExpandDims"))
        # custom pad so that no timepoint gets input from future
        NN.add(ZeroPadding1D(padding=(filt_size - 1, 0), name="ZeroPadding"))
        # Perform convolution (no pad)
        NN.add(Conv1D(filters=hidden_dim, kernel_size=filt_size,
                      padding='valid', activation=hidden_nonlin, name="Conv"))
        # squeeze dims for dense layers
        NN.add(Lambda(lambda x: tf.squeeze(x, [0]), name="Squeeze"))
    for i in range(num_layers):
        if is_shooter and add_pklayers:
            if row_sparse:
                PK_bias = PKRowBiasLayer(NN, srng, PKLparams)
            else:
                PK_bias = PKBiasLayer(NN, srng, PKLparams)
            PKbias_layers.append(PK_bias)
            NN = PK_bias
        if i == num_layers - 1:
            layer_dim = output_dim
            layer_nonlin = output_nonlin
        else:
            layer_dim = hidden_dim
            layer_nonlin = hidden_nonlin

        if batchnorm and i < num_layers - 1 and i != 0:
            NN.add(Dense(layer_dim,
                         name="Dense%s" % i,
                         kernel_initializer=tf.random_normal_initializer(
                            stddev=init_std)))
            NN.add(BatchNormalization(name="BatchNorm%s" % i))
            # may set initializer for hyperparams
            NN.add(Activation(activation=layer_nonlin,
                              name="Activation%s" % i))
        else:
            NN.add(Dense(layer_dim,
                         name="Dense%s" % i,
                         kernel_initializer=tf.random_normal_initializer(
                            stddev=init_std)))
            NN.add(Activation(activation=layer_nonlin,
                              name="Activation%s" % i))
    if add_pklayers:
        return NN, PKbias_layers
    else:
        return NN

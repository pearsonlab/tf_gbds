import tensorflow as tf
from tensorflow.contrib.keras import models
from tf_gbds.nn_utils import get_network


batch_size = 256
input_dim = 64
output_dim = 8
(hidden_dim, num_layers) = (32, 4)
filt_size = 16
params = {'a': 1, 'b': 1, 'k': 1}
num_biases = 4
data = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_dim))


def test_nn_utils_bias():
    NN, PKbias_layers = get_network(
        batch_size, input_dim, output_dim, hidden_dim, num_layers,
        PKLparams=params, srng=None, batchnorm=True, is_shooter=True,
        row_sparse=False, add_pklayers=True, filt_size=filt_size)
    assert isinstance(NN, models.Model)
    assert len(PKbias_layers) == num_layers

    params_count = ((filt_size * input_dim + 1) * hidden_dim +
                    num_biases * hidden_dim * 2 * num_layers +
                    hidden_dim * 4 * (num_layers - 2) +
                    (hidden_dim + 1) * hidden_dim * (num_layers - 1) +
                    (hidden_dim + 1) * output_dim)
    # ConvolutionalLayer + PKBiasLayers + BatchNormLayer(s) + DenseLayers
    assert NN.count_params() == params_count

    pad_shape = tf.TensorShape.as_list(
        NN.get_layer("ZeroPadding").output.shape)
    assert pad_shape == [1, batch_size + filt_size - 1, input_dim]

    conv_shape = tf.TensorShape.as_list(NN.get_layer("Conv").output.shape)
    assert conv_shape == [1, batch_size, hidden_dim]

    output = NN(data)
    output_shape = tf.TensorShape.as_list(output.shape)
    assert output_shape == [batch_size, output_dim]


def test_nn_utils_rowbias():
    NN, PKbias_layers = get_network(
        batch_size, input_dim, output_dim, hidden_dim, num_layers,
        PKLparams=params, srng=None, batchnorm=True, is_shooter=True,
        row_sparse=True, add_pklayers=True, filt_size=filt_size)
    assert isinstance(NN, models.Model)
    assert len(PKbias_layers) == num_layers

    params_count = ((filt_size * input_dim + 1) * hidden_dim +
                    num_biases * hidden_dim * 2 * num_layers +
                    hidden_dim * 4 * (num_layers - 2) +
                    (hidden_dim + 1) * hidden_dim * (num_layers - 1) +
                    (hidden_dim + 1) * output_dim)
    # ConvolutionalLayer + PKRowBiasLayers + BatchNormLayer(s) + DenseLayers
    assert NN.count_params() == params_count

    pad_shape = tf.TensorShape.as_list(
        NN.get_layer("ZeroPadding").output.shape)
    assert pad_shape == [1, batch_size + filt_size - 1, input_dim]

    conv_shape = tf.TensorShape.as_list(NN.get_layer("Conv").output.shape)
    assert conv_shape == [1, batch_size, hidden_dim]

    output = NN(data)
    output_shape = tf.TensorShape.as_list(output.shape)
    assert output_shape == [batch_size, output_dim]

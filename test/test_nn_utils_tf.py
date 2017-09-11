import tensorflow as tf
from tensorflow.contrib.keras import models
from tf_gbds.nn_utils_tf import get_network


batch_size = 256
input_dim = 64
output_dim = 8
(hidden_dim, num_layers) = (32, 4)
filt_size = 16
data = tf.random_normal((batch_size, input_dim), dtype=tf.float32)


def test_nn_utils():
    NN = get_network(
        batch_size, input_dim, output_dim, hidden_dim, num_layers,
        batchnorm=True, filt_size=filt_size)
    assert isinstance(NN, models.Model)

    pad_shape = tf.TensorShape.as_list(
        NN.get_layer("ZeroPadding").output.shape)
    assert pad_shape == [1, batch_size + filt_size - 1, input_dim]

    conv_shape = tf.TensorShape.as_list(NN.get_layer("Conv").output.shape)
    assert conv_shape == [1, batch_size, hidden_dim]

    output = NN(data)
    output_shape = tf.TensorShape.as_list(output.shape)
    assert output_shape == [batch_size, output_dim]

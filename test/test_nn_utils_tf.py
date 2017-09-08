import numpy as np
import numpy.testing as npt
import tensorflow as tf
from keras import layers, models
from nn_utils_tf import get_network
from layers import PKBiasLayer, PKRowBiasLayer


batch_size = 32
input_dim = 64
output_dim = 8
(hidden_dim, num_layers) = (16, 4)
filt_size = 4
data = tf.random_normal((batch_size, input_dim), dtype=tf.float32)

def test_nn_utils():
	NN = get_network(batch_size, input_dim, output_dim, hidden_dim, num_layers)
	output = NN(data)
	output_shape = tf.TensorShape.as_list(output.shape)

	assert isinstance(NN, models.Model)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		assert output_shape == [batch_size, output_dim]

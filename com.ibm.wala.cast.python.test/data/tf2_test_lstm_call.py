import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, layers

# Mirrors the `LSTM.call` recurrent model from
# `aymericdamien/TensorFlow-Examples/.../3_NeuralNetworks/recurrent_network.py`,
# a real-world sequence-classification utility (an LSTM layer followed by a dense
# read-out), for tensor-type inference coverage. Unlike the pure-`Dense`
# `NeuralNet.call` (`testNeuralNetwork*`), the forward pass begins with a
# built-in `tf.keras.layers.LSTM` over a rank-3 `(batch, timesteps, features)`
# input.
num_units = 32
num_classes = 10


class LSTM(Model):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm_layer = layers.LSTM(units=num_units)
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = self.lstm_layer(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x


lstm_net = LSTM()
x = tf.constant(np.ones((256, 28, 28), dtype=np.float32))
result = lstm_net(x, is_training=True)
assert x.shape == (256, 28, 28) and x.dtype == tf.float32
assert result.shape == (256, num_classes) and result.dtype == tf.float32

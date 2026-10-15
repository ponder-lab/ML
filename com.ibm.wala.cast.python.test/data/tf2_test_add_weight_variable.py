# Test for https://github.com/wala/ML/issues/1068.
import tensorflow as tf


def weight_sink(w):
    assert isinstance(w, tf.Variable)
    assert w.shape == (3, 2)
    assert w.dtype == tf.float32
    return w


class Weights(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[input_shape[-1], 2])

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def map_weights(self):
        return weight_sink(self.kernel)


layer = Weights()
layer(tf.ones([1, 3]))
layer.map_weights()

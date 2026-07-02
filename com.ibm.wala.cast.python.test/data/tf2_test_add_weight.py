# Probe for wala/ML#618: `self.add_weight(...)` (the Keras weight-creation API, called from the
# lazily-invoked `build`, wala/ML#595) creates a value the analysis classifies as a tensor, so a
# forward pass computing with the weight types.
import tensorflow as tf


def consume(t):
    pass


class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[4, 4], dtype="float32", initializer=tf.zeros_initializer()
        )

    def call(self, inputs):
        h = tf.linalg.matmul(inputs, self.kernel)
        consume(h)
        return h


layer = MyLayer()
out = layer(tf.ones((4, 4)))
assert out.shape == (4, 4)
assert out.dtype == tf.float32

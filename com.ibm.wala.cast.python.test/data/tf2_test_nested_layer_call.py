# Test for wala/ML#570: the result of a nested layer call (`self.inner(...)` inside another
# layer's `call`) carries the inner layer's forward-result type.
import tensorflow as tf


def consume(t):
    pass


class Inner(tf.keras.layers.Layer):
    def __init__(self):
        super(Inner, self).__init__()

    def call(self, inputs):
        return tf.linalg.matmul(inputs, inputs)


class Outer(tf.keras.layers.Layer):
    def __init__(self):
        super(Outer, self).__init__()
        self.inner = Inner()

    def call(self, inputs):
        x = self.inner(inputs)
        consume(x)
        return x


outer = Outer()
out = outer(tf.ones((4, 4)))
assert out.shape == (4, 4)
assert out.dtype == tf.float32

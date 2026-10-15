# Test for https://github.com/wala/ML/issues/1073.
import tensorflow as tf


class Pad(tf.keras.layers.Layer):
    def call(self, x):
        assert x.shape == (2, 5, 4)
        assert x.dtype == tf.float32
        return tf.pad(x, [[0, 0], [1, 1], [0, 0]])


class UnrolledPad(tf.keras.layers.Layer):
    def call(self, x):
        assert x.shape == (2, 5, 4)
        assert x.dtype == tf.float32
        return tf.pad(x, [[0, 0], [1, 1], [0, 0]])


class Looped(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.blocks = [tf.keras.layers.LeakyReLU(), Pad()]

    def call(self, x):
        y = x
        for layer in self.blocks:
            y = layer(y)
        return y


class Unrolled(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.activation = tf.keras.layers.LeakyReLU()
        self.pad = UnrolledPad()

    def call(self, x):
        y = self.activation(x)
        return self.pad(y)


assert Looped()(tf.ones((2, 5, 4))).shape == (2, 7, 4)
assert Unrolled()(tf.ones((2, 5, 4))).shape == (2, 7, 4)

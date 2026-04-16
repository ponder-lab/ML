import tensorflow as tf


class M(tf.keras.Model):
    def __init__(self):
        super(M, self).__init__()
        self.d = tf.keras.layers.Dense(4)

    def call(self, x):
        assert x.shape == (None, 3)
        assert x.dtype == tf.float32
        y = self.d(x)
        assert y.shape == (None, 4)
        assert y.dtype == tf.float32
        return y


inputs = tf.keras.Input(shape=(3,))
assert inputs.shape == (None, 3)
assert inputs.dtype == tf.float32

result = M()(inputs)
assert result.shape == (None, 4)
assert result.dtype == tf.float32
